import numpy as np
import tensorflow as tf

from pathlib import Path
from sklearn.preprocessing import StandardScaler
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

FILES_PATH = Path('./saved_models/')

np.random.seed(1)

kernel_initializer = tf.truncated_normal_initializer(stddev=0.1)


class StateScaler:
    '''
    This class created to scale and normalize the state
    It subtracts the mean and normalizes states to a unit variance.
    '''

    def __init__(self, env, num_samples=5000):
        state_samples = np.array([env.observation_space.sample() for _ in range(num_samples)])
        self.scaler = StandardScaler()
        self.scaler.fit(state_samples)

    def scale(self, state):
        '''
        input shape = (2,)
        output shape = (1,2)
        '''
        vector_len = state.shape[1]
        scaled = self.scaler.transform([state[0, 0:2].reshape([2])])
        new_state = np.pad(scaled.reshape(-1), (0, vector_len - 2))
        new_state = new_state.reshape([1, vector_len])
        return new_state


class ActorNetworkRegression:
    def __init__(self, state_size, action_size, learning_rate, scaler, session, env, num_of_h_layers=3,
                 name='actor_network'):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.scaler = scaler
        self.sess = session
        self.num_of_units = 12
        self.num_of_h_layers = num_of_h_layers
        self.name = name
        self.env = env
        with tf.variable_scope(name):
            self.state = tf.placeholder(tf.float32, [None, self.state_size], name="state")
            self.action = tf.placeholder(tf.float32, name="action")
            self.R_t = tf.placeholder(tf.float32, name="total_rewards")
            self.input = tf.layers.dense(units=self.num_of_units, inputs=self.state, activation=tf.nn.relu,
                                         kernel_initializer=kernel_initializer, name='input')
            self.h_layers = []

            for l in range(1, self.num_of_h_layers + 1):
                name_layer = 'h' + str(l)
                if l == 1:  # handle the first hidden layer, because of inputs.
                    if self.num_of_h_layers == 1:
                        h_f = tf.layers.dense(units=self.num_of_units, inputs=self.input, activation=tf.nn.relu,
                                              kernel_initializer=kernel_initializer, name=name_layer)
                    else:
                        h_f = tf.layers.dense(units=self.num_of_units, inputs=self.input,
                                              kernel_initializer=kernel_initializer, name=name_layer)

                else:  # handle the second and the rest h layer, because of inputs.
                    if self.num_of_h_layers == l:
                        h_f = tf.layers.dense(units=self.num_of_units, inputs=self.h_layers[len(self.h_layers) - 1],
                                              activation=tf.nn.relu,
                                              kernel_initializer=kernel_initializer, name=name_layer)
                    else:
                        h_f = tf.layers.dense(units=self.num_of_units, inputs=self.h_layers[len(self.h_layers) - 1],
                                              kernel_initializer=kernel_initializer, name=name_layer)
                self.h_layers.append(h_f)

            mu = tf.layers.dense(self.h_layers[len(self.h_layers) - 1], 1, None, kernel_initializer)
            sigma_t = tf.layers.dense(self.h_layers[len(self.h_layers) - 1], 1, None, kernel_initializer)
            sigma = tf.nn.softplus(sigma_t) + 1e-5

            norm_dist = tf.contrib.distributions.Normal(mu, sigma)
            action_t = tf.squeeze(norm_dist.sample(1), axis=0)
            self.action_clapped = tf.clip_by_value(action_t, np.array([-1]), np.array([1]))

            self.loss = -tf.log(norm_dist.prob(self.action) + 1e-5) * self.R_t
            self.optimizer = tf.train.AdamOptimizer(learning_rate).minimize(self.loss)

    def save_actor_model(self):
        full_path = str(FILES_PATH / self.env)
        for l in range(1, self.num_of_h_layers + 1):
            layer_name = 'h' + str(l)
            actor_bias = self.sess.run(
                tf.get_default_graph().get_tensor_by_name(self.name + "/" + layer_name + "/bias:0"))
            np.savetxt(full_path + "_actor_" + layer_name + "_bias.npy", actor_bias, delimiter=',')
            actor_weights = self.sess.run(
                tf.get_default_graph().get_tensor_by_name(self.name + "/" + layer_name + "/kernel:0"))
            np.savetxt(full_path + "_actor_" + layer_name + "_weights.npy", actor_weights, delimiter=',')
        print("saved new actor model!")
