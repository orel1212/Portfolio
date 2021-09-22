# Meta learning actor-critic algorithm for 3 gym enviroments
1. 'Acrobot-v1' -> 'CartPole-v1' <br>
2. 'CartPole-v1' ->'MountainCarContinuous-v0' <br>

## Results
1. 'Acrobot-v1' -> 'CartPole-v1' <br>
![alt tag](https://github.com/orel1212/MyWorks/blob/main/Reinforcement%20Learning/Policy_gradients_Reinforce/%E2%80%8F%E2%80%8Freinforce_results.PNG)
2. 'Acrobot-v1' <br>
![alt tag](https://github.com/orel1212/MyWorks/blob/main/Reinforcement%20Learning/Policy_gradients_Reinforce/%E2%80%8F%E2%80%8Freinforce_results.PNG)
3. 'CartPole-v1' ->'MountainCarContinuous-v0' <br>
![alt tag](https://github.com/orel1212/MyWorks/blob/main/Reinforcement%20Learning/Policy_gradients_Reinforce/%E2%80%8F%E2%80%8Freinforce_results.PNG)

### Comparison with base results
![alt tag](https://github.com/orel1212/MyWorks/blob/main/Reinforcement%20Learning/Policy_gradients_Reinforce/%E2%80%8F%E2%80%8Freinforce_results.PNG)
We can see that the fine tuning led to a faster convergence. The convergence was  <br>
significantly faster in terms of running times in both models. It also took the models less  <br>
episodes in order to reach convergence. MountainCarContinuous-v0 convergence was  <br>
much more stable than in the base ones. <br>

## Files
1. ac_meta.py ->  main script <br>
1. actor.py - actor implementation <br>
1. critic.py - discrete critic implementation ( 'CartPole-v1' + 'Acrobot-v1') <br>
1. actor_regression.py - continuous critic implementation ( 'MountainCarContinuous-v0') <br>

## How To Run
3. Fill in "game_list" in ac_meta.py the name of the env you would like to run
4. Run the script

## Requirements:
1. Python 3.7 <br>
2. Tensorflow 1.4 <br>

# Written by Orel Lavie and Noam Tzukerman

