
import pybullet_envs
import gym
import numpy as np
import torch
from sac_agent import SACAgent as Agent
import matplotlib.pyplot as plt


load_agent_flag = False
save_agent_flag = True
train_flag = True
render_flag = True

num_hidden_layers = 1
buffer_sample_size = 512
hidden_size = 64
alpha_actor = 3e-4
alpha_others = 3e-4
tau = 0.01
gamma = 0.9999
reward_scaling = 2

episodes = 600
num_steps = 1000
last_episodes_to_mean = 100

if __name__ == '__main__':

    env_id = 'InvertedPendulum-v2'
    env = gym.make(env_id)
    state_shape = env.observation_space.shape[0] #assume 1D input Env
    num_actions = env.action_space.shape[0]
    action_tanh_mult = env.action_space.high


    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')


    agent = Agent(device, alpha_actor = alpha_actor, alpha_others = alpha_others, input_dims = state_shape, num_actions = num_actions,action_tanh_mult = action_tanh_mult,
    tau = tau, gamma=gamma, num_hidden_layers = num_hidden_layers, hidden_size = hidden_size, buffer_sample_size = buffer_sample_size, scaling = reward_scaling)

    if load_agent_flag:
        agent.load()

    if render_flag:
        env.reset()
        env.render(mode='human')

    current_total_rewards = max(0,env.reward_range[0]) #start with min val
    rewards_history = []
    mean_rewards_history = []
    print(f'Env:{env_id}')

    for episode in range(1,episodes+1):
        state = env.reset()
        done = False
        total_rewards = 0
        steps = 1
        while not done and steps < num_steps:
            action = agent.act(state)
            state_, reward, done, info = env.step(action)
            steps += 1
            agent.store_in_buffer(state, action, reward, state_, done)
            if train_flag:
                agent.train()
            total_rewards += reward
            state = state_

        rewards_history.append(total_rewards)
        mean_episodes_num = min(last_episodes_to_mean,episode)
        mean_rewards = np.mean(rewards_history[-mean_episodes_num:])
        if episode >= mean_episodes_num:
            mean_rewards_history.append(mean_rewards)
        if mean_rewards > current_total_rewards:
            current_total_rewards = mean_rewards
            if save_agent_flag:
                agent.save()

        print(f'episode:{episode}, current_rewards:{total_rewards:.2f}, steps:{steps}')
        print(f'Last:{last_episodes_to_mean} episodes mean rewards:{mean_rewards:.2f}')

    plt.rcParams["figure.figsize"] = (16, 5)
    plt.plot(rewards_history, zorder=1, label='Rewards')
    plt.plot(list(range(99, len(mean_rewards_history) + 99)), mean_rewards_history, zorder=2, label='Mean over 100 episodes')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('Reward per episode')
    plt.legend()
    plt.savefig(f"rewards_per_ep_{len(rewards_history)}.png")
    plt.show()