import random
import time
from collections import deque

import gym
import numpy as np
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
from keras.utils.vis_utils import plot_model

num_of_steps = 500

num_of_episodes = 500

num_of_test_episodes = 1

env_req_shape = (1, 4)

exploit_mode_threshold = 475


class Model():
    def __init__(self, state_space, action_space, lr, n_hidden, hidden_size):
        self.input_space = state_space
        self.output_space = action_space
        self.n_hidden = n_hidden
        self.hidden_size = hidden_size
        self.lr = lr
        self.model = self.create_model()

    def create_model(self):
        model = Sequential()
        model.add(Dense(self.hidden_size, activation='relu', input_dim=self.input_space))
        for i in range(1, self.n_hidden):
            model.add(Dense(self.hidden_size, activation="relu"))
        model.add(Dense(self.output_space, activation='linear'))
        model.compile(optimizer=Adam(lr=self.lr), loss='mse', metrics=['accuracy'])
        plot_model(model, to_file=f'{self.n_hidden}_layers_model.png', show_shapes=True, show_layer_names=False)
        model.summary()
        return model


class DQN_Agent():
    def __init__(self, env,
                 learning_rate,
                 episodes,
                 discount_factor,
                 epsilon,
                 decay_rate,
                 minibatch_size,
                 experience_replay_size,
                 n_hidden,
                 hidden_size,
                 num_steps_to_update_network,
                 min_replay_size):
        self.num_steps_to_update_network = num_steps_to_update_network
        self.min_replay_size = min_replay_size
        self.env = env
        self.state_space = env.observation_space.shape[0]
        self.action_space = env.action_space.n
        self.helper_model = Model(self.state_space, self.action_space, learning_rate, n_hidden, hidden_size)
        self.target_model = Model(self.state_space, self.action_space, learning_rate, n_hidden, hidden_size)
        self.target_model.model.set_weights(self.helper_model.model.get_weights())
        self.experience_replay_buffer = deque(maxlen=experience_replay_size)
        self.episodes = episodes
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.decay_rate = decay_rate
        self.rewards_list = []
        self.loss_list = []
        self.minibatch_size = minibatch_size

    def train_minibatch(self):
        if len(self.experience_replay_buffer) <= self.min_replay_size:
            return 0

        minibatch = random.sample(self.experience_replay_buffer, self.minibatch_size)
        list_x = []
        list_y = []

        for index, (state, action, reward, next_state, done) in enumerate(minibatch):
            if done == True:
                predicted_y = reward
            else:
                qval_max_next_state = np.max(self.target_model.model.predict(next_state)[0])
                predicted_y = reward + (self.discount_factor * qval_max_next_state)

            state_predictions = self.helper_model.model.predict(state)
            state_predictions[0][action] = predicted_y
            list_x.append(state[0])
            list_y.append(state_predictions[0])

        list_x = np.array(list_x)
        list_y = np.array(list_y)

        result = self.helper_model.model.fit(list_x, list_y, epochs=1, verbose=0, batch_size=self.minibatch_size)
        curr_loss = result.history['loss'][0]
        return curr_loss

    def train(self):
        update_steps = 0
        for episode in range(self.episodes):
            state = self.env.reset()
            state = np.reshape(state, env_req_shape)
            start_episode_time = time.time()
            episode_reward = 0
            for step in range(num_of_steps):
                self.epsilon = self.epsilon * self.decay_rate

                if random.uniform(0, 1) < self.epsilon:
                    action = self.env.action_space.sample()
                else:
                    action = np.argmax(self.helper_model.model.predict(state)[0])

                new_state, reward, done, _ = self.env.step(action)  # take action, and take new step
                new_state = np.reshape(new_state, env_req_shape)
                episode_reward += reward

                self.experience_replay_buffer.append((state, action, reward, new_state, done))

                loss = self.train_minibatch()
                self.loss_list.append(loss)
                if update_steps > self.num_steps_to_update_network:
                    self.target_model.model.set_weights(self.helper_model.model.get_weights())
                    update_steps = 0

                update_steps += 1

                if done == True:
                    print("finished episode " + str(episode + 1) + " at step: " + str(step + 1))
                    break
                else:
                    state = new_state
            self.rewards_list.append(episode_reward)
            end_episode_time = time.time()
            if len(self.rewards_list) >= 100:
                episods_mean = np.mean(self.rewards_list[-100:])
                if episods_mean >= 475:
                    print(
                        f"At episode number :{episode} the agent first obtains an average reward of at least 475 over 100 consecutive episodes")
            print(f"Episode:{episode}, time: {end_episode_time - start_episode_time}, reward: {episode_reward}")
        print("total reward after training:")
        print(sum(self.rewards_list))
        return

    def test_agent(self, episodes):
        for episode in range(episodes):
            state = self.env.reset()
            state = np.reshape(state, env_req_shape)
            finished_successfully = False
            for i in range(num_of_steps):
                action = np.argmax(self.first_model.model.predict(state)[0])
                state, reward, done, _ = self.env.step(action)
                state = np.reshape(state, env_req_shape)
                self.env.render()
                if done == 1:
                    print("finished episode " + str(episode + 1) + " at step: " + str(i + 1))
                    finished_successfully = True
                    break
            if not finished_successfully:
                print("episode " + str(episode + 1) + " not finished, stopped at step 100")
        self.env.close()


env = gym.make('CartPole-v1')

np.random.seed(133)
env.seed(133)

agent_3 = DQN_Agent(env=env,
                    learning_rate=0.01,
                    episodes=num_of_episodes,
                    discount_factor=0.99,
                    epsilon=1,
                    decay_rate=0.99,
                    minibatch_size=8,
                    experience_replay_size=2048,
                    n_hidden=3,
                    hidden_size=16,
                    num_steps_to_update_network=32,
                    min_replay_size=100)

np.random.seed(120)
env.seed(120)

agent_5 = DQN_Agent(env=env,
                    learning_rate=0.01,
                    episodes=num_of_episodes,
                    discount_factor=0.99,
                    epsilon=1,
                    decay_rate=0.99,
                    minibatch_size=8,
                    experience_replay_size=2048,
                    n_hidden=5,
                    hidden_size=16,
                    num_steps_to_update_network=32,
                    min_replay_size=100)

agent_3.train()
agent_5.train()
agents = [agent_3, agent_5]

for agent in agents:
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    axes[0].plot(agent.rewards_list, zorder=1)
    axes[0].set_xlabel("Episode")
    axes[0].set_ylabel("Reward")
    axes[0].set_title(f"Reward per episode (with {agent.target_model.n_hidden} hidden layers)")

    rewards_100_episodes_averages = []
    range_obj = range(100, len(agent.rewards_list))
    for i in range_obj:
        rewards_100_episodes_averages.append(np.mean(agent.rewards_list[i - 100:i]))
    axes[0].plot(range_obj, rewards_100_episodes_averages, zorder=2)

    axes[1].plot(agent.loss_list)
    axes[1].set_xlabel("Step")
    axes[1].set_ylabel("Loss")
    axes[1].set_title(f"Loss per training step ({agent.target_model.n_hidden} hidden layers)")

    plt.savefig(f"statistics_dqn_{agent.target_model.n_hidden}.png")
    plt.show()
