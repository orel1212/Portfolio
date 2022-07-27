import random
import gym
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

num_of_steps = 100
num_of_episodes = 5000


class Qlearning():
    def __init__(self, env=gym.make('FrozenLake-v0'),
                 learning_rate=0.1,
                 episodes=num_of_episodes,
                 discount_factor=0.99,
                 epsilon=0.20,
                 decay_rate=0.9995
                 ):
        self.env = env
        self.epsilon = epsilon
        self.decay_rate = decay_rate
        self.learning_rate = learning_rate
        self.episodes = episodes
        self.discount_factor = discount_factor
        self.q_table = np.zeros(shape=(env.observation_space.n, env.action_space.n))  # shape - states, actions

    def train(self):
        saved_q_funcs, rewards, num_steps = [], [], []
        for episode in range(self.episodes):
            if episode == 500 or episode == 2000:
                saved_q_funcs.append(self.q_table.copy())
            state = self.env.reset()
            self.epsilon = self.epsilon * self.decay_rate

            for step in range(num_of_steps):

                if random.uniform(0, 1) < self.epsilon:
                    action = self.env.action_space.sample()

                else:
                    take_max_action = np.where(self.q_table[state] == self.q_table[state].max())[0]
                    action = np.random.choice(take_max_action)

                qval = self.q_table[state, action]

                new_state, reward, done, info = self.env.step(action)

                if done:
                    target = reward
                else:
                    max_next_state = np.max(self.q_table[new_state])
                    target = reward + (self.discount_factor * max_next_state)

                self.q_table[state, action] = (1 - self.learning_rate) * qval + self.learning_rate * target

                if done:
                    break
                else:
                    state = new_state

            rewards.append(target)
            num_steps.append(step if target != 0 else num_of_steps)

        saved_q_funcs.append(self.q_table.copy())
        return saved_q_funcs, rewards, num_steps


env = gym.make('FrozenLake-v0')
np.random.seed(1)
env.seed(1)

agent = Qlearning(env=env,
                  learning_rate=0.1,
                  episodes=num_of_episodes,
                  discount_factor=0.99,
                  epsilon=0.15,
                  decay_rate=0.9995)

saved_q_funcs, rewards, num_steps = agent.train()

for i in range(len(saved_q_funcs)):
    ax = sns.heatmap(saved_q_funcs[i], linewidth=0.1, annot=True, cmap="Reds")
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    plt.savefig(f"heatmap_qtable{i}.png")
    plt.show()

plt.plot(rewards)
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.title("Reward per episode")
plt.savefig("reward_per_episode.png")
plt.show()

x = list(range(0, num_of_episodes, num_of_steps))
plt.plot(x, [np.mean(num_steps[i:i + num_of_steps]) for i in x])
plt.xlabel("Number of episodes")
plt.ylabel("Number of steps")
plt.title("Average num of steps to the goal per episode")
plt.savefig("avg_steps_to_goal.png")
plt.show()
