# Policy gradients algorithms implementation for 'CartPole-v1' gym environment
## Reinforce
### Results (loss in each training step and the total reward of each episode)
![alt tag](https://github.com/orel1212/MyWorks/blob/main/Reinforcement%20Learning/Policy_gradients_Reinforce/%E2%80%8F%E2%80%8Freinforce_results.PNG)
### How To Run
run ./policy_gradients.py
## Reinforce with baseline (V func)
### Results (loss in each training step and the total reward of each episode)
![alt tag](https://github.com/orel1212/MyWorks/blob/main/Reinforcement%20Learning/Policy_gradients_Reinforce/%E2%80%8F%E2%80%8Freinforce_baseline_results.PNG)
### How To Run
run ./policy_gradients_with_baseline.py
## Actor Critic
### Results (loss in each training step and the total reward of each episode)
![alt tag](https://github.com/orel1212/MyWorks/blob/main/Reinforcement%20Learning/Policy_gradients_Reinforce/%E2%80%8F%E2%80%8Factor_critic.PNG)
### How To Run
run ./actor_critic.py

## Comparison
Conversion to an average reward of at least 475 over 100 consecutive episodes: <br>
1. Reinforce: 2633 episodes <br>
2. Reinforce with baseline: 1227 episodes <br>
3. Actor-critic: 1029 episodes <br>

We can see that REINFORCE is very unstable and it takes much more time to  <br>
converge than the other algorithms. Both REINFORCE with baseline and actor-critic,  <br>
converged much faster. However, actor-critic converged slightly faster, while being  <br>
much more stable compared to REINFORCE with baseline, which had many <br>
“exploration drops” throughout the way until it finally converged. <br>

# Requirements
● Python 3.7 <br>
● Tensorflow 1.4 <br>

# Written by Orel Lavie and Noam Tzukerman



