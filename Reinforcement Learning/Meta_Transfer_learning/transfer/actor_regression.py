import numpy as np
import tensorflow as tf

from pathlib import Path
from sklearn.preprocessing import StandardScaler
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

FILES_PATH = Path('./saved_models_tl/')

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
        new_state = np.pad(scaled.reshape(-1), (0, vector_len - 2), mode='constant')
        new_state = new_state.reshape([1, vector_len])
        return new_state


class ActorNetworkRegression:
    def __init__(self, state_size, action_size, learning_rate, scaler, session, env, envs_to_load, num_of_h_layers=1,
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
        self.envs_to_load = envs_to_load
        with tf.variable_scope(name):
            self.state = tf.placeholder(tf.float32, [None, self.state_size], name="state")
            self.action = tf.placeholder(tf.float32, name="action")
            self.R_t = tf.placeholder(tf.float32, name="total_rewards")
            self.input = tf.layers.dense(units=self.num_of_units, inputs=self.state, activation=None,
                                         kernel_initializer=kernel_initializer, name='input')
            saved_models = self.load_actor_model()
            self.old_models_layers = {}
            print("***")
            print(self.name)
            for env in self.envs_to_load:
                old_model = saved_models[env]
                self.old_models_layers[env] = []
                index_env_layers = 0
                for l in range(1, self.num_of_h_layers + 1):
                    layer_name = env + '_h' + str(l)
                    weights = old_model[l - 1][0]
                    biases = old_model[l - 1][1]
                    if len(self.envs_to_load) == 1 or self.envs_to_load[0] == env:  # need to load just 1 model
                        if l == 1:
                            activation = None
                            if self.num_of_h_layers == 1:
                                activation = tf.nn.relu
                            h = tf.layers.dense(units=self.num_of_units, inputs=self.input, activation=activation,
                                                kernel_initializer=weights,
                                                bias_initializer=biases, name=layer_name, trainable=False)
                        else:
                            activation = None
                            if self.num_of_h_layers == l:
                                activation = tf.nn.relu
                            h = tf.layers.dense(units=self.num_of_units,
                                                inputs=self.old_models_layers[env][index_env_layers],
                                                activation=activation,
                                                kernel_initializer=weights,
                                                bias_initializer=biases, name=layer_name, trainable=False)
                            print("input:" + str(self.old_models_layers[env][index_env_layers]))
                            index_env_layers += 1
                    elif self.envs_to_load[1] == env:  # if second env, first env must be input to second env
                        first_env = self.envs_to_load[0]
                        if l == 1:
                            activation = None
                            if self.num_of_h_layers == 1:
                                activation = tf.nn.relu
                            h = tf.layers.dense(units=self.num_of_units, inputs=self.input, activation=activation,
                                                kernel_initializer=weights,
                                                bias_initializer=biases, name=layer_name, trainable=False)
                        else:
                            inputs = [self.old_models_layers[first_env][index_env_layers],
                                      self.old_models_layers[env][index_env_layers]]
                            concatinated_inputs = tf.concat(inputs, 0)  # need to concat input before using
                            activation = None
                            if self.num_of_h_layers == l:
                                activation = tf.nn.relu
                            h = tf.layers.dense(units=self.num_of_units, inputs=concatinated_inputs,
                                                activation=activation,
                                                kernel_initializer=weights,
                                                bias_initializer=biases, name=layer_name, trainable=False)
                            print("input1:" + str(self.old_models_layers[first_env][index_env_layers]))
                            print("input2:" + str(self.old_models_layers[env][index_env_layers]))
                            index_env_layers += 1

                    self.old_models_layers[env].append(h)
                    print("output:" + str(h))

            print("***")
            self.h_layers = []
            index_env = -1  # somehow takes -1 instead of 0, idk why!!
            for l in range(1, self.num_of_h_layers + 1):

                name_layer = 'h' + str(l)
                if l == 1:  # handle the first hidden layer, because of inputs.
                    activation = None
                    if self.num_of_h_layers == 1:
                        activation = tf.nn.relu
                    h_f = tf.layers.dense(units=self.num_of_units, inputs=self.input, activation=activation,
                                          kernel_initializer=kernel_initializer, name=name_layer)

                else:  # handle the second and the rest h layer, because of inputs.
                    inputs = []
                    for env in self.envs_to_load:
                        old_inp = self.old_models_layers[env][index_env]
                        print("input:" + str(old_inp))
                        inputs.append(old_inp)  # add 1 layer before in previous models as input

                    curr_inp = self.h_layers[len(self.h_layers) - 1]
                    print("input:" + str(curr_inp))
                    inputs.append(curr_inp)  # last layer in current model
                    concatinated_inputs = tf.concat(inputs, 0)  # need to concat input before using
                    activation = None
                    if self.num_of_h_layers == l:
                        activation = tf.nn.relu
                    h_f = tf.layers.dense(units=self.num_of_units, inputs=concatinated_inputs, activation=activation,
                                          kernel_initializer=kernel_initializer, name=name_layer)
                self.h_layers.append(h_f)
                print("output:" + str(h_f))
                index_env += 1

            print("***")
            # creates last layer, output layer
            inputs = []
            for env in self.envs_to_load:
                index = len(self.old_models_layers[env]) - 1
                old_inp = self.old_models_layers[env][index]
                print("input:" + str(old_inp))
                inputs.append(old_inp)

            curr_inp = self.h_layers[len(self.h_layers) - 1]
            print("input:" + str(curr_inp))
            inputs.append(curr_inp)  # last layer in current model
            concatinated_inputs = tf.concat(inputs, 0)  # need to concat input before using
            print("***")

            mu = tf.layers.dense(concatinated_inputs, 1, None, kernel_initializer)
            sigma_t = tf.layers.dense(concatinated_inputs, 1, None, kernel_initializer)
            sigma = tf.nn.softplus(sigma_t) + 1e-5

            norm_dist = tf.contrib.distributions.Normal(mu, sigma)
            action_t = tf.squeeze(norm_dist.sample(1), axis=0)
            self.action_clapped = tf.clip_by_value(action_t, np.array([-1]), np.array([1]))

            self.loss = -tf.log(norm_dist.prob(self.action) + 1e-5) * self.R_t
            self.optimizer = tf.train.AdamOptimizer(learning_rate).minimize(self.loss)

    def load_actor_model(self):
        saved_models = {}
        for env in self.envs_to_load:
            full_path = str(FILES_PATH / env)
            h_layers = []
            for l in range(1, self.num_of_h_layers + 1):
                layer_name = 'h' + str(l)
                actor_bias = np.loadtxt(full_path + "_actor_" + layer_name + "_bias_section2.npy", delimiter=',')
                actor_weights = np.loadtxt(full_path + "_actor_" + layer_name + "_weights_section2.npy", delimiter=',')
                bias = tf.constant_initializer(actor_bias)
                weights = tf.constant_initializer(actor_weights)
                h = [weights, bias]
                h_layers.append(h)
            saved_models[env] = h_layers
        print("loaded an actor model!")
        return saved_models

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
