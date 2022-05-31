import numpy as np
import torch
import torch.nn as nn
from ac_agent import Agent
from buffer import Buffer
from env_wrappers import make_env
import matplotlib.pyplot as plt

output_file_path =  './learning_graph.png'

def create_worker(name, n_hidden, input_shape, n_actions, global_agent,optimizer, env_id, num_episodes ,max_timesteps):
    worker = Worker(name, n_hidden)
    worker.run(input_shape, n_actions, global_agent,optimizer, env_id, num_episodes ,max_timesteps)

class Worker:
    def __init__(self, name = '1', n_hidden = 256, clipping_rate = 40):
        self.name = name
        self.clipping_rate = clipping_rate
        self.n_hidden = n_hidden

    def plot_curve(self,episodes, total_rewards):
        running_avg = np.zeros(len(total_rewards))
        for i in range(len(running_avg)):
            running_avg[i] = np.mean(total_rewards[max(0, i - 100):(i + 1)])
        plt.plot(episodes, running_avg)
        plt.title('Average rewards (100 episodes average)')
        plt.savefig(output_file_path)

    def run(self,input_shape, n_actions, global_agent,
               optimizer, env_id, num_episodes ,max_timesteps):

        local_agent = Agent(input_shape, n_actions, self.n_hidden)

        buffer = Buffer()

        frame_buffer = [input_shape[1], input_shape[2], 1]
        env = make_env(env_id, shape=frame_buffer)

        episode, t_steps, total_rewards = 0, 0, []
        while episode < num_episodes:
            print(episode)
            obs = env.reset()
            total_reward, done, ep_steps = 0, False, 0
            hx = torch.zeros(1, self.n_hidden)
            while not done:
                state = torch.tensor(np.array([obs]), dtype=torch.float)
                action, v, log_prob, hx = local_agent(state, hx)
                obs_, reward, done, info = env.step(action)
                buffer.insert(obs, action, reward, obs_, v, log_prob)
                total_reward += reward
                obs = obs_
                ep_steps += 1
                t_steps += 1
                if done or ep_steps % max_timesteps == 0:
                    states, actions, rewards, new_states, v_data, log_probs = \
                            buffer.get_buffer()

                    loss = local_agent.loss_calc(obs, hx, done, rewards,
                                                 v_data, log_probs)
                    optimizer.zero_grad()
                    hx = hx.detach_()
                    #print(loss)
                    loss.backward()
                    nn.utils.clip_grad_norm_(local_agent.parameters(), self.clipping_rate)
                    for local_param, global_param in zip(
                                            local_agent.parameters(),
                                            global_agent.parameters()):
                        global_param._grad = local_param.grad
                    optimizer.step()
                    local_agent.load_state_dict(global_agent.state_dict())

                    buffer.clear()
            episode += 1
            if self.name != '1':
                continue
            else:
                total_rewards.append(total_reward)
                avg_reward = np.mean(total_rewards[-100:])
                print(f'Episode {episode}, Steps {t_steps}. Reward is {total_reward:.2f}')

                print(f'Average last 100 episodes reward {avg_reward:.2f}')

        if self.name != '1':
            return

        episodes = [e for e in range(episode)]
        self.plot_curve(episodes, total_rewards)