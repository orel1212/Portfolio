import numpy as np
import tensorflow as tf

from pathlib import Path

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

FILES_PATH = Path('./saved_models/')
np.random.seed(1)

kernel_initializer = tf.truncated_normal_initializer(stddev=0.1)


class ActorNetwork:
    def __init__(self, state_size, action_size, learning_rate, session, env, to_load=False, env_to_load='',
                 num_of_h_layers=1, name='actor_network'):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.sess = session
        self.num_of_units = 12
        self.num_of_h_layers = num_of_h_layers
        self.name = name
        self.env = env
        with tf.variable_scope(name):
            self.state = tf.placeholder(tf.float32, [None, self.state_size], name="state")
            self.action = tf.placeholder(tf.int32, [self.action_size], name="action")
            self.R_t = tf.placeholder(tf.float32, name="total_rewards")
            self.input = tf.layers.dense(units=self.num_of_units, inputs=self.state,
                                         kernel_initializer=kernel_initializer, name='input')
            if not to_load:
                self.h_layers = []

                for l in range(1, self.num_of_h_layers + 1):
                    name_layer = 'h' + str(l)
                    if l == 1:  # handle the first hidden layer, because of inputs.
                        activation = None
                        if self.num_of_h_layers == 1:
                            activation = tf.nn.relu
                        h_f = tf.layers.dense(units=self.num_of_units, inputs=self.input, activation=activation,
                                              kernel_initializer=kernel_initializer, name=name_layer)

                    else:  # handle the second and the rest h layer, because of inputs.
                        activation = None
                        if self.num_of_h_layers == l:
                            activation = tf.nn.relu
                        h_f = tf.layers.dense(units=self.num_of_units, inputs=self.h_layers[len(self.h_layers) - 1],
                                              activation=activation,
                                              kernel_initializer=kernel_initializer, name=name_layer)
                    self.h_layers.append(h_f)
            else:
                self.env_to_load = env_to_load
                self.load_actor_model()

            self.output = tf.layers.dense(inputs=self.h_layers[len(self.h_layers) - 1],
                                          units=self.action_size,
                                          kernel_initializer=kernel_initializer)

            # Softmax probability distribution over actions
            self.actions_distribution = tf.squeeze(tf.nn.softmax(self.output))
            # Loss with negative log probability
            self.neg_log_prob = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.output, labels=self.action)
            self.loss = tf.reduce_mean(self.neg_log_prob * self.R_t)
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)

    def load_actor_model(self):

        full_path = str(FILES_PATH / self.env_to_load)
        self.h_layers = []
        for l in range(1, self.num_of_h_layers + 1):
            layer_name = 'h' + str(l)
            actor_bias = np.loadtxt(full_path + "_actor_" + layer_name + "_bias.npy", delimiter=',')
            actor_weights = np.loadtxt(full_path + "_actor_" + layer_name + "_weights.npy", delimiter=',')
            bias = tf.constant_initializer(actor_bias)
            weights = tf.constant_initializer(actor_weights)
            if l == 1:
                activation = None
                if self.num_of_h_layers == 1:
                    activation = tf.nn.relu
                h = tf.layers.dense(units=self.num_of_units, inputs=self.input, activation=activation,
                                    kernel_initializer=weights,
                                    bias_initializer=bias, name=layer_name)
            else:
                activation = None
                if self.num_of_h_layers == l:
                    activation = tf.nn.relu
                h = tf.layers.dense(units=self.num_of_units, inputs=self.h_layers[len(self.h_layers) - 1],
                                    activation=activation,
                                    kernel_initializer=weights,
                                    bias_initializer=bias, name=layer_name)
            self.h_layers.append(h)
        print("loaded an actor model!")

    def save_actor_model(self):
        full_path = str(FILES_PATH / self.env)
        for l in range(1, self.num_of_h_layers + 1):
            layer_name = 'h' + str(l)
            actor_bias = self.sess.run(
                tf.get_default_graph().get_tensor_by_name(self.name + "/" + layer_name + "/bias:0"))
            np.savetxt(full_path + "_actor_" + layer_name + "_bias_section2.npy", actor_bias, delimiter=',')
            actor_weights = self.sess.run(
                tf.get_default_graph().get_tensor_by_name(self.name + "/" + layer_name + "/kernel:0"))
            np.savetxt(full_path + "_actor_" + layer_name + "_weights_section2.npy", actor_weights, delimiter=',')
        print("saved new actor model!")
