import gymnasium as gym
import os
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import A2C, DDPG, PPO, SAC
import imageio

import argparse


def record_video(model, video_filename, env_name="Ant-v5",num_episodes=1):
    """Record video of trained model playing the environment"""

    # Create environment for recording
    env = gym.make(env_name, render_mode='rgb_array')

    frames = []
    #all_eps_reward = []

    for episode in range(num_episodes):
        obs, info = env.reset()
        done = False
        truncated = False
        #episode_reward = 0

        while not (done or truncated):
            # Render frame
            frame = env.render()
            frames.append(frame)

            # Get action from model
            action, _ = model.predict(obs, deterministic=True)

            # Take step in environment
            obs, reward, done, truncated, info = env.step(action)
            #episode_reward += reward

        #print(f"Episode {episode + 1} reward: {episode_reward:.2f}")
        #all_eps_reward.append(episode_reward)

    env.close()

    # Save video
    imageio.mimsave(video_filename, frames, fps=30)
    print(f"Video saved as {video_filename}")



def evaluate_all_models(algorithms, models, env_name="Ant-v5", n_eval_episodes=50):
    evaluation_env = gym.make(env_name)
    results = {}

    for alg, model in zip(algorithms, models):
        print(f"\nEvaluating {alg} ....")
        
        rewards = []
        for _ in range(n_eval_episodes):
            obs, _ = evaluation_env.reset()
            done = False 
            total_reward = 0

            while not done: 
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, _ = evaluation_env.step(action)
                total_reward += reward
                done = terminated or truncated 

            rewards.append(total_reward)

        # Statistical Calculations
        data = np.array(rewards)
        mean_reward = data.mean()
        std_reward = data.std(ddof=1)
        
        # 95% Confidence Interval
        ci = 1.96 * std_reward / np.sqrt(len(data))
        
        # IQR Calculation
        q1 = np.percentile(data, 25)
        q3 = np.percentile(data, 75)

        # Store all metrics in the results dictionary
        results[alg] = {
            'mean': data.mean(),
            'ci': ci,
            'median': np.median(data),
            'q1': q1,
            'q3': q3,
            'iqr': q3 - q1,
            'min': data.min(),
            'max': data.max()}
    evaluation_env.close()
    return results


def printout(eval_results, algorithms, timesteps):
    print("\n" + "=" * 60)
    print("TRAINING SUMMARY")
    print("=" * 60)
    print(f"Environment: Ant-v5")
    print(f"Training timesteps: {timesteps}")
    print(f"Algorithms trained: {', '.join(algorithms)}")
    print("\nFinal Performance Over 50 episodes:")

    print(f"{'Algorithm':<12} | {'Mean ± 95% CI':<18} | {'Median':>10} | {'IQR':>10} | {'Min / Max':>20}")
    print("-" * 80)
    for alg in algorithms:
        res = eval_results[alg]
        stats = f"{res['mean']:>8.2f} ± {res['ci']:>6.2f}"
        range_str = f"{res['min']:.1f} / {res['max']:.1f}"
        
        print(f"{alg:<12} | {stats:<18} | {res['median']:>10.2f} | {res['iqr']:>10.2f} | {range_str:>20}")

    print("\nAll models have been trained and evaluated successfully!")

def patch_recording(algorithms, models, file_dir):
    for i in range(len(models)):
        record_video(models[i], os.path.join(file_dir, f"{algorithms[i]}.mp4"))




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

    VIDEO = "./videos/"
    os.makedirs(VIDEO, exist_ok=True)

    if 1 in args.exp:
        VIDEO_DIR = "./videos/experiment_A/"
        os.makedirs(VIDEO_DIR, exist_ok=True)
        algorithms = ['PPO-1000', 'PPO-4000']
        models = []
        timesteps = "1000_000 vs. 4000_000"
        print("Experiment A: Loading trained models...")

        # Load PPO model
        model_ppo_loaded = PPO.load("./models/experiment-A_ppo_ant_1000000_42")
        models.append(model_ppo_loaded)

        model_ppo_loaded = PPO.load("./models/experiment-A_ppo_ant_4000000_42")
        models.append(model_ppo_loaded)
        eval_results = evaluate_all_models(algorithms, models)
        printout(eval_results, algorithms,timesteps)

        print("=======generating record videos for sample in 1 episodes=======")
        patch_recording(algorithms, models, VIDEO_DIR)

    if 2 in args.exp:
        VIDEO_DIR = "./videos/experiment_B/"
        os.makedirs(VIDEO_DIR, exist_ok=True)

        algorithms = ['PPO (3e-4)', 'PPO (1.5e-3)', 'PPO (3e-3)']
        models = []
        timesteps = "1000_000"
        print("Experiment B: Loading trained models...")

                # Load PPO model
        model_ppo_loaded = PPO.load("./models/experiment-B_ppo_ant_0.0003_42")
        models.append(model_ppo_loaded)

        model_ppo_loaded = PPO.load("./models/experiment-B_ppo_ant_0.0015_42")
        models.append(model_ppo_loaded)

        model_ppo_loaded = PPO.load("./models/experiment-B_ppo_ant_0.003_42")
        models.append(model_ppo_loaded)

        eval_results = evaluate_all_models(algorithms, models)

        printout(eval_results, algorithms,timesteps)

        print("=======generating record videos for sample in 1 episodes=======")
        patch_recording(algorithms, models, VIDEO_DIR)


    if 3 in args.exp:
        VIDEO_DIR = "./videos/experiment_C/"
        os.makedirs(VIDEO_DIR, exist_ok=True)
        algorithms = ['A2C_500', 'DDPG_500', 'PPO_500', 'SAC_500']
        models = []
        timesteps = "500_000"
        print("Experiment C: Loading trained models...")
        model_a2c_loaded = A2C.load("./models/experiment-C_a2c_ant_500000_42")
        models.append(model_a2c_loaded)

        # Load DDPG model
        model_ddpg_loaded = DDPG.load("./models/experiment-C_ddpg_ant_500000_42")
        models.append(model_ddpg_loaded)

        # Load PPO model
        model_ppo_loaded = PPO.load("./models/experiment-C_ppo_ant_500000_42")
        models.append(model_ppo_loaded)


        # Load SAC model
        model_sac_loaded = SAC.load("./models/experiment-C_sac_ant_500000_42")
        models.append(model_sac_loaded)

        eval_results = evaluate_all_models(algorithms, models)
        printout(eval_results, algorithms,timesteps)

        print("=======generating record videos for sample in 1 episodes=======")
        patch_recording(algorithms, models, VIDEO_DIR)
#############################3########################################################################
        algorithms1 = ['A2C_1000', 'DDPG_1000', 'PPO_1000', 'SAC_1000']
        models1 = []
        timesteps = "1000_000"
        print("Experiment C: Loading trained models...")
        model_a2c_loaded = A2C.load("./models/experiment-C_a2c_ant_1000000_42")
        models1.append(model_a2c_loaded)

        # Load DDPG model
        model_ddpg_loaded = DDPG.load("./models/experiment-C_ddpg_ant_1000000_42")
        models1.append(model_ddpg_loaded)

        # Load PPO model
        model_ppo_loaded = PPO.load("./models/experiment-C_ppo_ant_1000000_42")
        models1.append(model_ppo_loaded)


        # Load SAC model
        model_sac_loaded = SAC.load("./models/experiment-C_sac_ant_1000000_42")
        models1.append(model_sac_loaded)

        eval_results = evaluate_all_models(algorithms1, models1)
        printout(eval_results, algorithms1,timesteps)
        print("=======generating record videos for sample in 1 episodes=======")
        patch_recording(algorithms1, models1, VIDEO_DIR)

