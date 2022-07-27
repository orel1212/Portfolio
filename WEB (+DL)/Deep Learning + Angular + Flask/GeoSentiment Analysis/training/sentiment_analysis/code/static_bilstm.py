import tensorflow as tf
import numpy as np
import datetime
import random
from config import ConfigLSTM

import os


# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 


class BiLSTM:
    """A BiLSTM class model"""

    def __init__(self):
        self.config = ConfigLSTM()

    def build_train_graph(self, word_vectors_shape):
        with tf.variable_scope("Model"):
            self.predictions = self.build_model_graph(word_vectors_shape)
            self.loss, self.accuracy = self.build_loss_graph()
            self.optimizer = self.build_optimizer_graph()
            with tf.variable_scope("Valid_loss"):
                self.min_valid_loss_to_save = tf.get_variable(name="min_valid_loss_to_save", trainable=False, shape=[],
                                                              dtype=tf.float32,
                                                              initializer=tf.constant_initializer(0.6))
                self.min_valid_loss_placeholder = tf.placeholder(dtype=tf.float32, name='min_valid_loss_placeholder')
                self.valid_assign_op = tf.assign(self.min_valid_loss_to_save, self.min_valid_loss_placeholder,
                                                 name="valid_assign_op")
        self.summary = self.build_summary_graph()

    def restore_train_graph(self, sess):

        graph = tf.get_default_graph()
        self.global_step = graph.get_tensor_by_name("Model/Global_step/global_step:0")
        self.increment_global_step_op = graph.get_tensor_by_name("Model/Global_step/inc_global_step_op:0")

        # TODO: CHECK ABOUT GET OPERATION
        self.optimizer = graph.get_operation_by_name("Model/Optimizer/optimizer")
        self.loss = graph.get_tensor_by_name("Model/Loss/loss:0")
        self.accuracy = graph.get_tensor_by_name("Model/Accuracy/accuracy:0")

        self.summary_placeholder = []
        self.summary_placeholder.append(graph.get_tensor_by_name("Summary/train_loss_placeholder:0"))
        self.summary_placeholder.append(graph.get_tensor_by_name("Summary/train_acc_placeholder:0"))
        self.summary_placeholder.append(graph.get_tensor_by_name("Summary/valid_loss_placeholder:0"))
        self.summary_placeholder.append(graph.get_tensor_by_name("Summary/valid_acc_placeholder:0"))

        self.summary = []
        self.summary.append(graph.get_tensor_by_name("Summary/train_loss:0"))
        self.summary.append(graph.get_tensor_by_name("Summary/train_accuracy:0"))
        self.summary.append(graph.get_tensor_by_name("Summary/valid_loss:0"))
        self.summary.append(graph.get_tensor_by_name("Summary/valid_accuracy:0"))

        self.valid_assign_op = graph.get_tensor_by_name("Model/Valid_loss/valid_assign_op:0")
        self.min_valid_loss_placeholder = graph.get_tensor_by_name("Model/Valid_loss/min_valid_loss_placeholder:0")
        self.min_valid_loss_to_save = graph.get_tensor_by_name("Model/Valid_loss/min_valid_loss_to_save:0")
        self.input_placeholder = graph.get_tensor_by_name("Model/Input/input:0")
        self.labels_placeholder = graph.get_tensor_by_name("Model/Input/labels:0")
        self.keep_prob_placeholder = graph.get_tensor_by_name("Model/Input/keep_prob:0")
        self.classification = graph.get_tensor_by_name("Model/BiLSTM/classification:0")

    def lstm_cell_with_dropout(self):
        return tf.contrib.rnn.DropoutWrapper(cell=self.lstm_cell(), output_keep_prob=self.keep_prob_placeholder,
                                             seed=self.config.dropout_seed)

    def lstm_cell(self):  # changed basicLSTMCELL to LSTMCELL
        return tf.contrib.rnn.LSTMCell(self.config.hidden_units)

    def build_model_graph(self, word_vectors_shape):
        with tf.name_scope("Input"):
            self.input_placeholder = tf.placeholder(dtype=tf.int32, shape=[None, self.config.timesteps], name='input')
            self.labels_placeholder = tf.placeholder(dtype=tf.int32, shape=[None, self.config.classes_num],
                                                     name='labels')
            self.keep_prob_placeholder = tf.placeholder(tf.float32, name='keep_prob')
        with tf.name_scope("Global_step"):
            self.global_step = tf.Variable(1, name='global_step', trainable=False, dtype=tf.int32)
            self.increment_global_step_op = tf.assign(self.global_step, self.global_step + 1, name="inc_global_step_op")

        # self.learning_rate=tf.train.exponential_decay(self.config.learning_rate, self.global_step,1, 0.99, staircase=False)
        self.learning_rate = tf.train.inverse_time_decay(self.config.learning_rate, self.global_step, 1, 1,
                                                         name="learning_rate")

        with tf.variable_scope("Embeddings"):
            embedding_weights = tf.get_variable(name="embedding_weights", shape=word_vectors_shape, trainable=False,
                                                initializer=tf.zeros_initializer())
            self.embedding_placeholder = tf.placeholder(tf.float32, word_vectors_shape)
            self.embedding_init = embedding_weights.assign(self.embedding_placeholder)
            embedding = tf.nn.embedding_lookup(embedding_weights, self.input_placeholder)
            inputs = [tf.squeeze(x, [1]) for x in tf.split(embedding, self.config.timesteps, 1)]

        with tf.variable_scope("BiLSTM"):
            if self.config.layers == 1:
                forward = self.lstm_cell_with_dropout()
                backward = self.lstm_cell_with_dropout()
            else:
                forward = tf.contrib.rnn.MultiRNNCell(
                    [self.lstm_cell_with_dropout() for _ in range(self.config.layers)])
                backward = tf.contrib.rnn.MultiRNNCell(
                    [self.lstm_cell_with_dropout() for _ in range(self.config.layers)])
            rnn_outputs, _, _ = tf.contrib.rnn.static_bidirectional_rnn(forward, backward, inputs, dtype=tf.float32)
            self.weights = tf.get_variable('weights', shape=[2 * self.config.hidden_units, self.config.classes_num],
                                           initializer=tf.contrib.layers.xavier_initializer(uniform=True, seed=50,
                                                                                            dtype=tf.float32))
            bias = tf.get_variable('bias', shape=[self.config.classes_num], initializer=tf.zeros_initializer())
            # predictions = (tf.matmul(rnn_outputs[-1], self.weights) + bias)
            predictions = tf.add(tf.matmul(rnn_outputs[-1], self.weights), bias, name="predictions")

            self.classification = tf.argmax(predictions, 1, name="classification")

        return predictions

    def build_loss_graph(self):
        with tf.name_scope('Accuracy'):
            correctPred = tf.equal(tf.argmax(self.predictions, 1), tf.argmax(self.labels_placeholder, 1))
            accuracy = tf.reduce_mean(tf.cast(correctPred, tf.float32), name="accuracy")
            # tf.nn.softmax_cross_entropy_with_logits IS DEPRECATED
        with tf.name_scope("Loss"):
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.predictions,
                                                                       labels=self.labels_placeholder)
            loss = tf.reduce_mean(cross_entropy)
            regularizer = tf.nn.l2_loss(self.weights)
            loss = tf.reduce_mean(loss + self.config.beta * regularizer, name="loss")
        return loss, accuracy

    def build_optimizer_graph(self):
        with tf.name_scope('Optimizer'):
            optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
            gradients, variables = zip(*optimizer.compute_gradients(loss=self.loss))
            gradients, _ = tf.clip_by_global_norm(gradients, 5.0)
            optimizer = optimizer.apply_gradients(zip(gradients, variables), name="optimizer")
            # optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss,global_step=self.global_step)
        return optimizer

    def build_summary_graph(self):
        with tf.name_scope('Summary'):
            train_loss_placeholder = tf.placeholder(tf.float32, name="train_loss_placeholder")
            train_acc_placeholder = tf.placeholder(tf.float32, name="train_acc_placeholder")
            valid_loss_placeholder = tf.placeholder(tf.float32, name="valid_loss_placeholder")
            valid_acc_placeholder = tf.placeholder(tf.float32, name="valid_acc_placeholder")
            self.summary_placeholder = [train_loss_placeholder, train_acc_placeholder, valid_loss_placeholder,
                                        valid_acc_placeholder]
            # f.summary.histogram("weights",self.weights)
            summary = []
            summary.append(tf.summary.scalar(name='train_loss', tensor=train_loss_placeholder))
            summary.append(tf.summary.scalar(name='train_accuracy', tensor=train_acc_placeholder))
            summary.append(tf.summary.scalar(name='valid_loss', tensor=valid_loss_placeholder))
            summary.append(tf.summary.scalar(name='valid_accuracy', tensor=valid_acc_placeholder))
            # summary.append(tf.summary.scalar(name='loss_difference', tensor=self.vector_placeholder))
        return summary

    def run_epoch(self, sess, batch_size, x, y, tensors, dropout, train_flag=True):
        combined_list = list(zip(x, y))
        random.shuffle(combined_list)
        x, y = zip(*combined_list)

        iter_num = len(x) // batch_size
        remaining = len(x) - iter_num * batch_size

        start_index = 0
        avg_loss = 0
        avg_acc = 0
        for i in range(iter_num):
            batch_inputs = x[start_index:start_index + batch_size]
            batch_labels = y[start_index:start_index + batch_size]
            feed_dict = {
                self.input_placeholder: batch_inputs,
                self.labels_placeholder: batch_labels,
                self.keep_prob_placeholder: dropout
            }
            if train_flag:
                _, loss_val, acc_val = sess.run(fetches=tensors, feed_dict=feed_dict)
            else:
                loss_val, acc_val = sess.run(fetches=tensors, feed_dict=feed_dict)
            start_index += batch_size
            avg_loss += loss_val
            avg_acc += acc_val

        if remaining > 0:
            batch_inputs = x[start_index:start_index + batch_size]
            batch_labels = y[start_index:start_index + batch_size]
            feed_dict = {
                self.input_placeholder: batch_inputs,
                self.labels_placeholder: batch_labels,
                self.keep_prob_placeholder: dropout
            }
            if train_flag:
                _, loss_val, acc_val = sess.run(fetches=tensors, feed_dict=feed_dict)
            else:
                loss_val, acc_val = sess.run(fetches=tensors, feed_dict=feed_dict)

            avg_loss = (avg_loss * batch_size + loss_val * remaining) / (batch_size * iter_num + remaining)
            avg_acc = (avg_acc * batch_size + acc_val * remaining) / (batch_size * iter_num + remaining)
        else:
            avg_loss /= iter_num
            avg_acc /= iter_num
        return avg_loss, avg_acc * 100


def update_summary(sess, sum_op, placeholder, value, ep, writer):
    summary = sess.run(sum_op, {placeholder: value})
    writer.add_summary(summary, ep)
    writer.flush()


def train(data_handler, bi_lstm, sess, writer, train_saver, valid_saver, model_train_path, model_valid_path):
    metric_tensors = [bi_lstm.loss, bi_lstm.accuracy]
    dropout_train = bi_lstm.config.dropout
    batch_size = bi_lstm.config.batch_size
    x_train = data_handler.ids_matrix["train_matrix"]
    y_train = data_handler.labels["train"]
    x_valid = data_handler.ids_matrix["valid_matrix"]
    y_valid = data_handler.labels["valid"]
    x_test = data_handler.ids_matrix["test_matrix"]
    y_test = data_handler.labels["test"]
    valid_diff = 0.025

    min_valid_loss = sess.run(bi_lstm.min_valid_loss_to_save)
    print("current min valid loss: ", min_valid_loss)
    print('Starting training!')
    for ep in range(bi_lstm.config.epochs_num):
        step = sess.run(bi_lstm.global_step)
        print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        train_avg_loss, train_avg_acc = bi_lstm.run_epoch(sess, batch_size, x_train, y_train,
                                                          [bi_lstm.optimizer] + metric_tensors, dropout_train)
        print("ep: ", step, ' train avg loss: ', train_avg_loss, 'train avg acc: ', train_avg_acc)
        valid_avg_loss, valid_avg_acc = bi_lstm.run_epoch(sess, batch_size, x_valid, y_valid, metric_tensors, 1, False)
        print("ep: ", step, ' valid avg loss: ', valid_avg_loss, 'valid avg acc: ', valid_avg_acc)

        update_summary(sess, bi_lstm.summary[0], bi_lstm.summary_placeholder[0], train_avg_loss, step, writer)
        update_summary(sess, bi_lstm.summary[1], bi_lstm.summary_placeholder[1], train_avg_acc, step, writer)
        update_summary(sess, bi_lstm.summary[2], bi_lstm.summary_placeholder[2], valid_avg_loss, step, writer)
        update_summary(sess, bi_lstm.summary[3], bi_lstm.summary_placeholder[3], valid_avg_acc, step, writer)

        # increment before saving to start from the new epoch
        step = sess.run(bi_lstm.increment_global_step_op)

        if (step - 1) % 30 == 0:
            valid_diff /= 2
        if min_valid_loss - valid_avg_loss > valid_diff:
            min_valid_loss = valid_avg_loss
            sess.run(bi_lstm.valid_assign_op, feed_dict={bi_lstm.min_valid_loss_placeholder: valid_avg_loss})
            valid_saver.save(sess, model_valid_path + "model.ckpt", global_step=step - 1)
            print("saved valid model")

        if (step - 1) % 10 == 0:
            train_saver.save(sess, model_train_path + "model.ckpt", global_step=step - 1)
            print("saved train model")

    test_avg_loss, test_avg_acc = bi_lstm.run_epoch(sess, batch_size, x_test, y_test, metric_tensors, 1, False)
    print("test avg loss: ", test_avg_loss, 'test avg acc: ', test_avg_acc)


def test(data_handler, bi_lstm, sess):
    metric_tensors = [bi_lstm.loss, bi_lstm.accuracy]
    x_test = data_handler.ids_matrix["test_matrix"]
    y_test = data_handler.labels["test"]
    test_avg_loss, test_avg_acc = bi_lstm.run_epoch(sess, bi_lstm.config.batch_size, x_test, y_test, metric_tensors, 1,
                                                    False)
    print("test avg loss: ", test_avg_loss, 'test avg acc: ', test_avg_acc)


def save_prediction_graph(sess, bi_lstm, export_path_base, model_version):
    export_path = export_path_base + str(model_version)
    print('Exporting trained model to', export_path)
    builder = tf.saved_model.builder.SavedModelBuilder(export_path)

    tensor_info_x = tf.saved_model.utils.build_tensor_info(bi_lstm.input_placeholder)
    tensor_info_dropout = tf.saved_model.utils.build_tensor_info(bi_lstm.keep_prob_placeholder)
    tensor_info_y = tf.saved_model.utils.build_tensor_info(bi_lstm.classification)

    prediction_signature = (
        tf.saved_model.signature_def_utils.build_signature_def(
            inputs={'tweets': tensor_info_x, 'dropout': tensor_info_dropout},
            outputs={'scores': tensor_info_y},
            method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME))

    legacy_init_op = tf.group(tf.tables_initializer(), name='legacy_init_op')

    builder.add_meta_graph_and_variables(
        sess, [tf.saved_model.tag_constants.SERVING],
        signature_def_map={
            'prediction':
                prediction_signature,
        },
        legacy_init_op=legacy_init_op)

    builder.save()
    print('Done exporting!')
