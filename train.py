import gymnasium as gym
import os
from stable_baselines3 import A2C, DDPG, PPO, SAC, TD3
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import load_results, ts2xy
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
from stable_baselines3.common.callbacks import BaseCallback, ProgressBarCallback
import numpy as np
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.utils import set_random_seed
import torch
import gc
import argparse



class TrainingCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(TrainingCallback, self).__init__(verbose)
        self.rewards = []
        self.episode_lengths = []

    def _on_step(self) -> bool:
        if len(self.locals.get('infos', [])) > 0:
            for info in self.locals['infos']:
                if 'episode' in info:
                    self.rewards.append(info['episode']['r'])
                    self.episode_lengths.append(info['episode']['l'])
        return True




def make_env(env_id, rank, seed,file):
    """
    Utility function for multiprocessed env.

    :param env_id: (str) the environment ID
    :param seed: (int) the inital seed for RNG
    :param rank: (int) index of the subprocess
    """

    def _init():
        env_ppo = Monitor(gym.make(env_id), LOG_DIR + f"{file}/")
        # use a seed for reproducibility
        # Important: use a different seed for each environment
        # otherwise they would generate the same experiences
        env_ppo.reset(seed=seed + rank)
        return env_ppo
    
    set_random_seed(seed)
    return _init


def PPO_train_rate(TOTAL_TIMESTEPS, seed, learning, experiment):
    env_name = "Ant-v5"
    env_ppo = make_vec_env(
        env_name, 
        n_envs=1, 
        seed=42, 
        monitor_dir=LOG_DIR + f"{experiment}/")

    # Initialize PPO model
    model_ppo = PPO(
        "MlpPolicy",
        env_ppo,
        verbose=0,
        tensorboard_log=LOG_DIR + f"{experiment}/",
        learning_rate=learning,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        clip_range_vf=None,
        ent_coef=0.0,
        vf_coef=0.5,
        max_grad_norm=0.5,
        use_sde=False,
        sde_sample_freq=-1,
        seed = seed,
        device="cpu"
    )
    
    # Train PPO
    callback_ppo = TrainingCallback()
    model_ppo.learn(total_timesteps=TOTAL_TIMESTEPS, callback=callback_ppo, progress_bar=True)
    model_ppo.save(f"./models/{experiment}_ppo_ant_{learning}_{seed}")
    
    # delete model and env.
    env_ppo.close()
    del model_ppo
    del env_ppo
    gc.collect()

    print("PPO training completed! ✅")


def PPO_train(TOTAL_TIMESTEPS, seed, cpu, experiment):
    num_cpu = cpu
    env_id = "Ant-v5"
    env_ppo = SubprocVecEnv([make_env(env_id, i, seed, f"{experiment}") for i in range(num_cpu)])
    print(f"Initalizing PPO model... with seed = {seed}")
    
    print(f"Training PPO... with {TOTAL_TIMESTEPS} in {cpu} environments")
    batchSize = 64 * cpu
    # Initialize PPO model
    model_ppo = PPO(
        "MlpPolicy",
        env_ppo,
        verbose=0,
        tensorboard_log=LOG_DIR + f"{experiment}/",
        learning_rate=0.0003,
        n_steps=2048,
        batch_size=batchSize,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        clip_range_vf=None,
        ent_coef=0.0,
        vf_coef=0.5,
        max_grad_norm=0.5,
        use_sde=False,
        sde_sample_freq=-1,
        seed = seed,
        device="cpu"
    )
    
    # Train PPO
    callback_ppo = TrainingCallback()
    model_ppo.learn(total_timesteps=TOTAL_TIMESTEPS, callback=callback_ppo, progress_bar=True)
    model_ppo.save(f"./models/{experiment}_ppo_ant_{TOTAL_TIMESTEPS}_{seed}")
    
    # delete model and env.
    env_ppo.close()
    del model_ppo
    del env_ppo
    gc.collect()

    print("PPO training completed! ✅")



    
def A2C_train(TOTAL_TIMESTEPS, seed, cpu, experiment):
    num_cpu = cpu
    env_id = "Ant-v5"
    env_a2c = SubprocVecEnv([make_env(env_id, i, seed, f"experiment") for i in range(num_cpu)])
    print(f"Initalizing A2C model... with seed = {seed}")
    
    print(f"Training A2C... with {TOTAL_TIMESTEPS} in {cpu} environments")

    # Initialize A2C model
    model_a2c = A2C(
        "MlpPolicy",
        env_a2c,
        verbose=0,
        tensorboard_log=LOG_DIR + f"{experiment}/",
        learning_rate=0.0003,
        n_steps=256, ## 256*4 = 1024
        gamma=0.99,
        gae_lambda=0.95,
        ent_coef=0.0,
        vf_coef=0.5,
        max_grad_norm=0.5,
        use_rms_prop=True,
        rms_prop_eps=1e-05,
        use_sde=False,
        sde_sample_freq=-1,
        normalize_advantage=False,
        seed = seed,
        device="cpu"
    )
    # Train A2C
    callback_a2c = TrainingCallback()
    model_a2c.learn(total_timesteps=TOTAL_TIMESTEPS, callback=callback_a2c, progress_bar=True)
    model_a2c.save(f"./models/{experiment}_a2c_ant_{TOTAL_TIMESTEPS}_{seed}")
    #video_path_a2c = os.path.join(VIDEO_DIR, "a2c_ant.mp4")
    #reward_a2c = record_video(model_a2c, env_name, video_path_a2c)

    ## delete and clean
    env_a2c.close()
    del model_a2c
    del env_a2c
    gc.collect()
    
    print("A2C training completed! ✅")


def DDPG_train(TOTAL_TIMESTEPS, seed, cpu, experiment):
    num_cpu = cpu
    env_id = "Ant-v5"
    env_ddpg = SubprocVecEnv([make_env(env_id, i, seed, f"{experiment}") for i in range(num_cpu)])
    print(f"Initalizing DDPG model...with seed = {seed}")
    
    print(f"Training DDPG... with {TOTAL_TIMESTEPS} in {cpu} environment")
    
    # Action noise for exploration
    n_actions = env_ddpg.action_space.shape[-1]
    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))
    
    # Initialize DDPG model
    model_ddpg = DDPG(
        "MlpPolicy",
        env_ddpg,
        action_noise=action_noise,
        verbose=0,
        tensorboard_log=LOG_DIR + f"{experiment}/",
        learning_rate=0.0003,
        buffer_size=300_000,
        learning_starts=100,
        batch_size=256,
        tau=0.005,
        gamma=0.99,
        train_freq=1,
        gradient_steps=2, ## upadate 2 among 4 envs. 
        seed = seed,
        device="auto"
    )
    
    # Train DDPG
    callback_ddpg = TrainingCallback()
    model_ddpg.learn(total_timesteps=TOTAL_TIMESTEPS, callback=callback_ddpg, progress_bar=True)
    model_ddpg.save(f"./models/{experiment}_ddpg_ant_{TOTAL_TIMESTEPS}_{seed}")

    ## delete and clean
    env_ddpg.close()
    del model_ddpg
    del env_ddpg
    gc.collect()
    
    print("DDPG training completed! ✅")


def SAC_train(TOTAL_TIMESTEPS, seed, cpu, experiment):
    num_cpu = cpu
    env_id = "Ant-v5"
    env_sac = SubprocVecEnv([make_env(env_id, i, seed, f"{experiment}") for i in range(num_cpu)])
    print(f"Initalizing SAC model... with seed = {seed}")
    
    print(f"Training SAC... with {TOTAL_TIMESTEPS} in {cpu} Environments")
    
    
    # Initialize SAC model
    model_sac = SAC(
        "MlpPolicy",
        env_sac,
        verbose=0,
        tensorboard_log=LOG_DIR + f"{experiment}/",
        learning_rate=0.0003,
        buffer_size=300_000,
        learning_starts=100,
        batch_size=256,
        tau=0.005,
        gamma=0.99,
        train_freq=1,
        gradient_steps=2,# 1
        ent_coef="auto",
        target_update_interval=1,
        target_entropy="auto",
        use_sde=False,
        sde_sample_freq=-1,
        use_sde_at_warmup=False,
        seed = seed,
        device="auto"
    )
    
    # Train SAC
    callback_sac = TrainingCallback()
    model_sac.learn(total_timesteps=TOTAL_TIMESTEPS, callback=callback_sac,progress_bar=True)
    model_sac.save(f"./models/{experiment}_sac_ant_{TOTAL_TIMESTEPS}_{seed}")

    ## delete and clean
    env_sac.close()
    del model_sac
    del env_sac
    gc.collect()
    
    print("SAC training completed! ✅")




if __name__ == "__main__":

    # 1. Initialize the parser 
    parser = argparse.ArgumentParser(description="Course: EE5329 RL Final Project\nTraining Quadruped Locomotion in Ant-v5 moving")

    parser.add_argument(
        "-e","--exp",
        type = int,
        nargs="+",
        choices= [1,2,3],
        required = True,
        help="Specify which experiment(s) to run: 1, 2, 3, or multiple (e.g, --exp 1)"
    )

    ## input 
    args = parser.parse_args()

    ## set up files and log 
    LOG_DIR = "./logs/"
    MODEL_DIR = "./models/"
    VIDEO_DIR = "./videos/"
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(VIDEO_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)
    

    ## execute experiment

    """
    first experiment 
    - Comparing Random Agent with PPO Agent baseline
    - Base on PPO, change `total timesteps`
    - PPO Agent is running with 1000,000 vs. 4000,000 timesteps, which 1000 vs. 4000 epsoides
    - seed = 42
    """
    if 1 in args.exp:
        cpu = 1
        seed = 42
        TOTAL_TIMESTEPS = 1000_000
        experiment = "experiment-A"
        print("Experiment A: PPO_1000 vs. PPO_4000")
        print("\n" + "=" * 60)
        PPO_train(TOTAL_TIMESTEPS, seed, cpu, experiment)
        TOTAL_TIMESTEPS = 4000_000
        PPO_train(TOTAL_TIMESTEPS, seed, cpu, experiment)
    
    if 2 in args.exp:

        TOTAL_TIMESTEPS = 1000_000
        seed = 42
        experiment = "experiment-B"
        learning = 0.0003
        print("\nExperiment B: PPO_1000 in different learning rate: 3e-4 | 1.5e-3 | 3e-3")
        print("\n" + "=" * 60)

        print(f"\nTraining PPO within 1000_000 timesteps: learning_rate = {learning}")
        PPO_train_rate(TOTAL_TIMESTEPS, seed, learning, experiment)

        learning = 0.0015
        print(f"Training PPO within 1000_000 timesteps: learning_rate = {learning}")
        PPO_train_rate(TOTAL_TIMESTEPS, seed, learning, experiment)

        learning = 0.003
        print(f"Training PPO within 1000_000 timesteps: learning_rate = {learning}")
        PPO_train_rate(TOTAL_TIMESTEPS, seed, learning, experiment)


    if 3 in args.exp:
        cpu = 4
        TOTAL_TIMESTEPS = 10_000
        seed = 42
        experiment = "experiment-C"

        print(f"Start Trainging: \nSeed List: {[seed+ x for x in range(4)]} within {TOTAL_TIMESTEPS} timesteps")
        print("--------On Policy------------")
        PPO_train(TOTAL_TIMESTEPS, seed, cpu, experiment)
        A2C_train(TOTAL_TIMESTEPS,seed,cpu, experiment)
        print("--------Off Policy------------")
        DDPG_train(TOTAL_TIMESTEPS,seed,cpu, experiment)
        SAC_train(TOTAL_TIMESTEPS,seed,cpu, experiment)

        
        TOTAL_TIMESTEPS = 50_000
        print(f"Start Trainging: \nSeed List: {[seed+ x for x in range(4)]} within {TOTAL_TIMESTEPS} timesteps")
        print("--------On Policy------------")
        PPO_train(TOTAL_TIMESTEPS, seed, cpu, experiment)
        A2C_train(TOTAL_TIMESTEPS,seed,cpu,experiment)
        print("--------Off Policy------------")
        DDPG_train(TOTAL_TIMESTEPS,seed,cpu,experiment)
        SAC_train(TOTAL_TIMESTEPS,seed,cpu,experiment)





    
    


