
from collections import deque

import gym
import numpy as np
import tensorflow as tf

import collections
import matplotlib.pyplot as plt

env = gym.make('CartPole-v1')

np.random.seed(1)

kernel_initializer = tf.contrib.layers.xavier_initializer(seed=0)

class ValueBaseline():
    def __init__(self, state_size, action_size, learning_rate, name='value_network'):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.output_size = 1
        self.units = 32

        with tf.variable_scope(name):

            self.state = tf.placeholder(tf.float32, [None, self.state_size], name="state")
            self.R_t = tf.placeholder(tf.float32, name="total_rewards")

            h1_inputs = tf.layers.dense(units=self.units, inputs=self.state, kernel_initializer=kernel_initializer,activation=tf.nn.relu)
            h2_inputs = tf.layers.dense(units=self.units, inputs=h1_inputs, kernel_initializer=kernel_initializer,activation=tf.nn.relu)
            h3_inputs = tf.layers.dense(units=self.units, inputs=h2_inputs, kernel_initializer=kernel_initializer,activation=tf.nn.relu)
            self.output = tf.layers.dense(units=self.output_size, inputs=h3_inputs, kernel_initializer=kernel_initializer,activation=None)


            # Value function estimation
            self.value_estimate = tf.squeeze(self.output)
            self.loss = tf.reduce_mean(tf.squared_difference(self.value_estimate, self.R_t))
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)



class PolicyNetwork:
    def __init__(self, state_size, action_size, learning_rate, name='policy_network'):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate

        with tf.variable_scope(name):

            self.state = tf.placeholder(tf.float32, [None, self.state_size], name="state")
            self.action = tf.placeholder(tf.int32, [self.action_size], name="action")
            self.R_t = tf.placeholder(tf.float32, name="total_rewards_with_baseline")

            self.W1 = tf.get_variable("W1", [self.state_size, 12], initializer=kernel_initializer)
            self.b1 = tf.get_variable("b1", [12], initializer=tf.zeros_initializer())
            self.W2 = tf.get_variable("W2", [12, self.action_size], initializer=kernel_initializer)
            self.b2 = tf.get_variable("b2", [self.action_size], initializer=tf.zeros_initializer())

            self.Z1 = tf.add(tf.matmul(self.state, self.W1), self.b1)
            self.A1 = tf.nn.relu(self.Z1)
            self.output = tf.add(tf.matmul(self.A1, self.W2), self.b2)

            # Softmax probability distribution over actions
            self.actions_distribution = tf.squeeze(tf.nn.softmax(self.output))
            # Loss with negative log probability
            self.neg_log_prob = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.output, labels=self.action)
            self.loss = tf.reduce_mean(self.neg_log_prob * self.R_t)
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)


# Define hyperparameters
state_size = 4
action_size = env.action_space.n

max_episodes = 5000
max_steps = 501
discount_factor = 0.99
policy_learning_rate = 0.005
value_learning_rate = 0.005
render = False

# Initialize the policy network
tf.reset_default_graph()
policy = PolicyNetwork(state_size, action_size, policy_learning_rate)
value_baseline = ValueBaseline(state_size, action_size, value_learning_rate)

# Start training the agent with REINFORCE_with_baseline algorithm
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    solved = False
    Transition = collections.namedtuple("Transition", ["state", "action", "reward", "next_state", "done"])
    episode_rewards = np.zeros(max_episodes)
    avg_episode_rewards = deque(maxlen=100)
    average_rewards = 0.0
    policy_loss = []
    value_loss = []
    rewards = []
    average_rewards_lst = []
    for episode in range(max_episodes):
        state = env.reset()
        state = state.reshape([1, state_size])
        episode_transitions = []

        for step in range(max_steps):
            actions_distribution = sess.run(policy.actions_distribution, {policy.state: state})
            action = np.random.choice(np.arange(len(actions_distribution)), p=actions_distribution)
            next_state, reward, done, _ = env.step(action)
            next_state = next_state.reshape([1, state_size])

            if render:
                env.render()

            action_one_hot = np.zeros(action_size)
            action_one_hot[action] = 1
            episode_transitions.append(Transition(state=state, action=action_one_hot, reward=reward, next_state=next_state, done=done))
            episode_rewards[episode] += reward

            if done:
                rewards.append(episode_rewards[episode])
                if episode > 98:
                    # Check if solved
                    average_rewards = np.mean(episode_rewards[(episode - 99):episode+1])
                    average_rewards_lst.append(average_rewards)
                    print("Episode {} Reward: {} Average over 100 episodes: {}".format(episode, episode_rewards[episode],round(average_rewards, 2)))
                else:
                    average_rewards_lst.append(0)

                if average_rewards > 475:
                    print(' Solved at episode: ' + str(episode))
                    solved = True
                break
            state = next_state

        if solved:
            break

        # Compute Rt for each time-step t and update the network's weights

        for t, transition in enumerate(episode_transitions):
            total_discounted_return = sum(discount_factor ** i * t.reward for i, t in enumerate(episode_transitions[t:])) # Rt

            # use baseline estimator and then update advantage
            baseline_value = sess.run(value_baseline.value_estimate, {value_baseline.state: state})


            # baseline update
            feed_dict_value = {value_baseline.state: transition.state, value_baseline.R_t: total_discounted_return}
            _, loss_value = sess.run([value_baseline.optimizer, value_baseline.loss], feed_dict_value)
            value_loss.append(loss_value)

            # advantage = episode_reward - baseline
            advantage_value = total_discounted_return - baseline_value


            feed_dict_policy = {policy.state: transition.state, policy.R_t: advantage_value, policy.action: transition.action}
            _, loss_policy = sess.run([policy.optimizer, policy.loss], feed_dict_policy)
            policy_loss.append(loss_policy)


    plt.rcParams["figure.figsize"] = (16, 5)
    plt.plot(rewards, zorder=1, label='Rewards')
    plt.plot(list(range(99, len(average_rewards_lst) + 99)), average_rewards_lst, zorder=2, label='Mean over 100 episodes')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('Reward per episode')
    plt.legend()
    plt.savefig(f"advantage_rewards_per_ep_{len(rewards)}.png")
    plt.show()
    plt.plot(value_loss, zorder=1, label='Value loss')
    plt.plot(policy_loss, zorder=2, label='Policy loss')
    plt.xlabel('Episode')
    plt.ylabel('Loss')
    plt.title('Policy Loss')
    plt.legend()
    plt.savefig(f"advantage_loss_{len(rewards)}.png")
    plt.show()

