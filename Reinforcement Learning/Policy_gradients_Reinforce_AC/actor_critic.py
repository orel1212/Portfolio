import gym
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

env = gym.make('CartPole-v1')
np.random.seed(1)
kernel_initializer = tf.truncated_normal_initializer(stddev=0.1)


class ActorNetwork:
    def __init__(self, state_size, action_size, learning_rate, name='actor_network'):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate

        with tf.variable_scope(name):
            self.state = tf.placeholder(tf.float32, [None, self.state_size], name="state")
            self.action = tf.placeholder(tf.int32, [self.action_size], name="action")
            self.R_t = tf.placeholder(tf.float32, name="total_rewards")

            self.W1 = tf.get_variable("W1", [self.state_size, 12], initializer=kernel_initializer)
            self.b1 = tf.get_variable("b1", [12], initializer=kernel_initializer)
            self.W2 = tf.get_variable("W2", [12, self.action_size], initializer=kernel_initializer)
            self.b2 = tf.get_variable("b2", [self.action_size], initializer=kernel_initializer)

            self.Z1 = tf.add(tf.matmul(self.state, self.W1), self.b1)
            self.A1 = tf.nn.relu(self.Z1)
            self.output = tf.add(tf.matmul(self.A1, self.W2), self.b2)

            # Softmax probability distribution over actions
            self.actions_distribution = tf.squeeze(tf.nn.softmax(self.output))
            # Loss with negative log probability
            self.neg_log_prob = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.output, labels=self.action)
            self.loss = tf.reduce_mean(self.neg_log_prob * self.R_t)
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)


class CriticNetwork:
    def __init__(self, state_size, learning_rate, num_of_layers=3, name='critic_network'):
        self.state_size = state_size
        self.learning_rate = learning_rate

        with tf.compat.v1.variable_scope(name):
            self.state = tf.compat.v1.placeholder(tf.float32, [None, self.state_size], name="state")
            self.R_t = tf.compat.v1.placeholder(tf.float32, name="total_rewards")

            self.hidden_layer = tf.layers.dense(inputs=tf.expand_dims(self.state, 0),
                                                units=32,
                                                activation=tf.nn.relu,
                                                kernel_initializer=kernel_initializer)

            for i in range(1, num_of_layers):
                h = tf.layers.dense(units=32,
                                    inputs=self.hidden_layer,
                                    kernel_initializer=kernel_initializer,
                                    activation=tf.nn.relu)

            self.output = tf.layers.dense(inputs=h,
                                          units=1,
                                          kernel_initializer=kernel_initializer)

            self.loss = tf.math.squared_difference(tf.squeeze(self.output), self.R_t)
            self.optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)


state_size = 4
action_size = env.action_space.n

max_episodes = 5000
max_steps = 501
discount_factor = 0.99
lr_actor = 0.0001
lr_critic = 0.001

render = False

# Initialize the policy network
tf.compat.v1.reset_default_graph()
tf.get_logger().setLevel('INFO')
actor = ActorNetwork(state_size, action_size, lr_actor)
critic = CriticNetwork(state_size, lr_critic)

rewards = []
rewards_avg = []
actor_loss = []
critic_loss = []

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    solved = False
    episode_rewards = np.zeros(max_episodes)
    average_rewards = 0.0

    for episode in range(max_episodes):
        state = env.reset()
        state = state.reshape([1, state_size])

        for step in range(max_steps):
            actions_distribution = sess.run(actor.actions_distribution, {actor.state: state})
            action = np.random.choice(np.arange(len(actions_distribution)), p=actions_distribution)
            next_state, reward, done, _ = env.step(action)
            next_state = next_state.reshape([1, state_size])

            if render:
                env.render()

            action_one_hot = np.zeros(action_size)
            action_one_hot[action] = 1
            episode_rewards[episode] += reward

            td_value = reward

            if not done:
                td_value += discount_factor * sess.run(critic.output, {critic.state: next_state})

            td_error = td_value - sess.run(critic.output, {critic.state: state})

            feed_dict = {critic.state: state, critic.R_t: td_value}
            _, loss = sess.run([critic.optimizer, critic.loss], feed_dict)
            critic_loss.append(loss)

            feed_dict = {actor.state: state,
                         actor.R_t: td_error,
                         actor.action: action_one_hot}
            _, loss = sess.run([actor.optimizer, actor.loss], feed_dict)
            actor_loss.append(loss)

            if done:
                rewards.append(episode_rewards[episode])

                if episode > 98:
                    # Check if solved
                    average_rewards = np.mean(episode_rewards[(episode - 99):episode + 1])
                    rewards_avg.append(average_rewards)
                print("Episode {} Reward: {} Average over 100 episodes: {}".format(episode, episode_rewards[episode],
                                                                                   round(average_rewards, 2)))
                if average_rewards >= 475:
                    print(' Solved at episode: ' + str(episode))
                    solved = True
                break
            state = next_state

        if solved:
            break

plt.rcParams["figure.figsize"] = (16, 5)
plt.plot(rewards, zorder=1, label='Rewards')
plt.plot(list(range(99, len(rewards_avg) + 99)), rewards_avg, zorder=2, label='Mean over 100 episodes')
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.title('Reward per episode')
plt.legend()
plt.savefig(f"rewards_ac_{len(rewards)}.png")
plt.show()

plt.plot(critic_loss, zorder=1, label='Critic loss')
plt.plot(actor_loss, zorder=2, label='Actor loss')
plt.xlabel('Episode')
plt.ylabel('Loss')
plt.title('Actor and Critic Losses')
plt.legend()
plt.savefig(f"loss_ac_{len(rewards)}.png")
plt.show()
