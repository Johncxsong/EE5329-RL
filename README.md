

# Training Quadruped Locomotion (Ant-v5 moving)  

- Name: John Song
- Course: EE5329
- Term: Spring 2026


## Quick Start  

### 1. Environment setup

1. open terminal run `conda create -n rlproj python=3.11 -y`
2. run `conda activate rlproj`
3. run `python -m pip install --upgrade pip`
4. run `pip install --no-cache-dir -r requirements.txt`

`chmod +x setup_rl.sh`  

`./setup_rl.sh`

### 2. Run code
1. `git clone https://github.com/Johncxsong/EE5329-RL.git` or download as zip file
2. run `cd `
3. run train part `python train.py --exp 1`
4. run evaluation part `python evaluation.py --exp 1`


### 3. Experiment explantion
- `train.py` contains: `experiment A`, `experiment B`, and `experiment C`
```python
python train.py --exp 1  # experiment A  (baseline)
python train.py --exp 2  # experiment B  (comparing learning rate)
python train.py --exp 3  # experiment C  (benchmark)

#### or run all three together
python train.py --exp 1 2 3
```

- `evaluation.py` contains: `experiment A`, `experiment B`, and `experiment C`  for each numerical result and recording videos. 

```python
python evaluation.py --exp 1  # experiment A  (baseline)
python evaluation.py --exp 2  # experiment B  (comparing learning rate)
python evaluation.py --exp 3  # experiment C  (benchmark)

#### or run all three together
python evaluation.py --exp 1 2 3
```


### 4. Check training process (learning curve, training loss...)

1. open terminal `cd `
2. run `tensorboard --logdir ./logs/experiment_A` to check epxeriment A training 




## Performance  
### Experiment A  








### Experiment B 







### Experiment C 






## Problems 

- use `--no-cache-dir` flag to install `pip install numpy --no-cache-dir`