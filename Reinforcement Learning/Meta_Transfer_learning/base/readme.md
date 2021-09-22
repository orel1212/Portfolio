# Base actor-critic algorithm for 3 gym enviroments
1. 'CartPole-v1' <br>
2. 'Acrobot-v1' <br>
3. 'MountainCarContinuous-v0' <br>

## Results
1. 'CartPole-v1' <br>
![alt tag](https://github.com/orel1212/MyWorks/blob/main/Reinforcement%20Learning/Meta_Transfer_learning/base/%E2%80%8F%E2%80%8Fcart_pole.PNG)
2. 'Acrobot-v1' <br>
![alt tag](https://github.com/orel1212/MyWorks/blob/main/Reinforcement%20Learning/Meta_Transfer_learning/base/%E2%80%8F%E2%80%8Facrobot.PNG)
3. 'MountainCarContinuous-v0' <br>
![alt tag](https://github.com/orel1212/MyWorks/blob/main/Reinforcement%20Learning/Meta_Transfer_learning/base/%E2%80%8F%E2%80%8Fmountain_car.PNG)

## Files
1. ac_base.py ->  main script <br>
1. actor.py - actor implementation <br>
1. critic.py - discrete critic implementation ( 'CartPole-v1' + 'Acrobot-v1') <br>
1. actor_regression.py - continuous critic implementation ( 'MountainCarContinuous-v0') <br>

## How To Run
3. Fill in "game_list" in ac_base.py the name of the env you would like to run
4. Run the script

## Requirements:
1. Python 3.7 <br>
2. Tensorflow 1.4 <br>

# Written by Orel Lavie and Noam Tzukerman

