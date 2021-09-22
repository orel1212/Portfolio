# Progessive Networks Transfer learning actor-critic algorithm for 3 gym enviroments
1. 'Acrobot-v1' -> 'MountainCarContinuous-v0' -> 'CartPole-v1' <br>
2. 'CartPole-v1' -> 'Acrobot-v1' -> 'MountainCarContinuous-v0' <br>

## Architecture
![alt tag](https://github.com/orel1212/MyWorks/blob/main/Reinforcement%20Learning/Meta_Transfer_learning/transfer/%E2%80%8F%E2%80%8Fprogressive_networks.PNG) <br>
Tp make the convergence work, we built the architecture as a progressive network.   <br>
First we train the first model, and then use the result as an input to the second model.  <br>We then use both models as an input to the third model layers.   <br>For
example, {acrobot, mountainCar}->cartpole means, we first train the acrobot model, 
then use the trained acrobot layers output as an input to train mountainCar,  <br> and then
use them both as an input to the cartpole layers.  <br>

## Results
1. 'Acrobot-v1' -> 'MountainCarContinuous-v0' -> 'CartPole-v1' <br>
![alt tag](https://github.com/orel1212/MyWorks/blob/main/Reinforcement%20Learning/Meta_Transfer_learning/transfer/acrobot_mountain_car_to_%E2%80%8F%E2%80%8Fcartpole.PNG)
2. 'CartPole-v1' -> 'Acrobot-v1' -> 'MountainCarContinuous-v0' <br>
![alt tag](https://github.com/orel1212/MyWorks/blob/main/Reinforcement%20Learning/Meta_Transfer_learning/transfer/%E2%80%8F%E2%80%8Fcartpole_acrobot_to_mountain_car.PNG)

### Comparison with base results
![alt tag](https://github.com/orel1212/MyWorks/blob/main/Reinforcement%20Learning/Meta_Transfer_learning/transfer/%E2%80%8F%E2%80%8Fcomparison.PNG) <br>
We can see that the convergence was faster in both models, but not significantly faster
in MountainCarContinuous-v0.   <br>The transfer learning however did improve the training
and it took the models less episode in order to reach convergence. <br>

## Files
1. ac_transfer.py ->  main script <br>
1. actor.py - actor implementation <br>
1. critic.py - discrete critic implementation ( 'CartPole-v1' + 'Acrobot-v1') <br>
1. actor_regression.py - continuous critic implementation ( 'MountainCarContinuous-v0') <br>

## How To Run
3. Fill in "game_list" in ac_transfer.py the name of the env you would like to run
4. Run the script

## Requirements:
1. Python 3.7 <br>
2. Tensorflow 1.4 <br>

# Written by Orel Lavie and Noam Tzukerman

