# DDQN algorithm implementation for 'CartPole-v1' gym environment
## Hyperparameters
1. learning rate - 0.01 <br>
2. discount factor - 0.99
3. epsilon decay rate - 0.99 ( exponential decay)

## Architecture
![alt tag](https://github.com/orel1212/MyWorks/blob/main/Reinforcement%20Learning/DQN/%E2%80%8F%E2%80%8Farchitectures.PNG)

## Results
At episode number 291 the agent first obtains an average reward of <br>
at least 475 over 100 consecutive episodes.

## the loss in each training step
![alt tag](https://github.com/orel1212/MyWorks/blob/main/Reinforcement%20Learning/DQN/%E2%80%8F%E2%80%8Floss_per_step.PNG)

## the total reward of each episode in training
![alt tag](https://github.com/orel1212/MyWorks/blob/main/Reinforcement%20Learning/DQN/%E2%80%8F%E2%80%8Frewards_per_episode.PNG)

# How To Run
run ./ddqn_cart_pole.py

