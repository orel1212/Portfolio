"""
Based on the paper: On Predicting Geolocation of Tweets using Convolutional Neural Network
by Binxuan Huang and Kathleen M. Carley
"""

import tensorflow as tf
import numpy as np
from config import ConfigCNN
import random


class CNN():
    def __init__(self):
        self.config = ConfigCNN()
        #self.build_train_graph()

    def restore_train_graph(self,sess):
        graph = tf.get_default_graph()
        self.global_step = graph.get_tensor_by_name("Model/Global_step/global_step:0")
        self.increment_global_step_op = graph.get_tensor_by_name("Model/Global_step/inc_global_step_op:0")

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

        self.input_content_placeholder = graph.get_tensor_by_name("Model/Input/input_content:0")
        self.input_description_placeholder = graph.get_tensor_by_name("Model/Input/input_description:0")
        self.input_username_placeholder = graph.get_tensor_by_name("Model/Input/input_username:0")
        self.input_profile_location_placeholder = graph.get_tensor_by_name("Model/Input/input_profile_location:0")
        self.input_misc_placeholder = graph.get_tensor_by_name("Model/Input/input_misc:0")
        self.input_depth_placeholder = graph.get_tensor_by_name("Model/Input/input_depth:0")
        self.labels_placeholder = graph.get_tensor_by_name("Model/Input/labels:0")
        self.keep_prob_placeholder = graph.get_tensor_by_name("Model/Input/keep_prob:0")
        self.classification = graph.get_tensor_by_name("Model/Output/classification:0")

    def build_train_graph(self, word_vectors_shape):
        with tf.variable_scope("Model"):
            with tf.name_scope("Input"):
                self.input_content_placeholder = tf.placeholder(tf.int32, [None, self.config.sequence_content], name="input_content")
                self.input_description_placeholder = tf.placeholder(tf.int32, [None, self.config.sequence_description], name="input_description")
                self.input_username_placeholder = tf.placeholder(tf.int32, [None, self.config.sequence_username], name="input_username")
                self.input_profile_location_placeholder = tf.placeholder(tf.int32, [None, self.config.sequence_profile_location], name="input_profile_location")
                self.input_misc_placeholder = tf.placeholder(tf.int32, [self.config.misc_size, None], name="input_misc")
                self.input_depth_placeholder = tf.placeholder(tf.int32, [5], name="input_depth")
                self.labels_placeholder = tf.placeholder(tf.int32, [None], name="labels")
                self.keep_prob_placeholder = tf.placeholder(tf.float32, name='keep_prob')

            with tf.variable_scope("Embeddings"):
                embedding_weights = tf.get_variable(name = "embedding_weights", shape = word_vectors_shape, trainable = False, initializer=tf.zeros_initializer())
                self.embedding_placeholder = tf.placeholder(tf.float32, shape = word_vectors_shape)
                self.embedding_init = embedding_weights.assign(self.embedding_placeholder)
                embed_input_list = [] 

                embedding = tf.nn.embedding_lookup(embedding_weights, self.input_content_placeholder)
                embedded_input_content = tf.expand_dims(embedding, -1)
                embed_input_list.append(embedded_input_content)

                embedding = tf.nn.embedding_lookup(embedding_weights, self.input_description_placeholder)
                embedded_input_description = tf.expand_dims(embedding, -1)
                embed_input_list.append(embedded_input_description)

                embedding = tf.nn.embedding_lookup(embedding_weights, self.input_username_placeholder)
                embedded_input_username = tf.expand_dims(embedding, -1)
                embed_input_list.append(embedded_input_username)

                embedding = tf.nn.embedding_lookup(embedding_weights, self.input_profile_location_placeholder)
                embedded_input_profile_location = tf.expand_dims(embedding, -1)
                embed_input_list.append(embedded_input_profile_location)

            with tf.variable_scope("Valid_loss"):
                self.min_valid_loss_to_save = tf.get_variable(name="min_valid_loss_to_save", trainable=False, shape = [], dtype=tf.float32, initializer = tf.constant_initializer(0.6))
                self.min_valid_loss_placeholder = tf.placeholder(dtype=tf.float32, name='min_valid_loss_placeholder')
                self.valid_assign_op = tf.assign(self.min_valid_loss_to_save, self.min_valid_loss_placeholder, name = "valid_assign_op")

            with tf.name_scope("Global_step"):
                self.global_step = tf.Variable(1, name='global_step', trainable=False, dtype=tf.int32)
                self.increment_global_step_op = tf.assign(self.global_step, self.global_step+1, name= "inc_global_step_op")


            with tf.variable_scope("One_hot_encoding"):

                tweet_lang = tf.one_hot(self.input_misc_placeholder[0], self.input_depth_placeholder[0])
                user_lang = tf.one_hot(self.input_misc_placeholder[1], self.input_depth_placeholder[1])
                time_zone = tf.one_hot(self.input_misc_placeholder[2], self.input_depth_placeholder[2])
                posting_timeslot = tf.one_hot(self.input_misc_placeholder[3], self.input_depth_placeholder[3])
                theta_misc = tf.concat([tweet_lang, user_lang, time_zone, posting_timeslot], 1)
                self.labels = tf.one_hot(self.labels_placeholder, self.input_depth_placeholder[4])

            def build_conv_layers(config, embed_input_list):

                output_dict = {"content" : [],
                               "description" : [],
                               "username" : [],
                               "profile_location" : []}

                with tf.name_scope("Conv_layer"):
                    for i, filter_size in enumerate(config.filter_sizes):
                        filter_shape = [filter_size, config.embed_dim, 1, config.filter_num]
                        filter_weights = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="filter_weights")
                        filter_bias = tf.Variable(tf.constant(0.1, shape=[config.filter_num]), name="filter_bias")
                        conv_content = tf.nn.conv2d(
                            embed_input_list[0],
                            filter_weights,
                            strides=[1, 1, 1, 1],
                            padding="VALID",
                            name="conv_content")

                        conv_description = tf.nn.conv2d(
                            embed_input_list[1],
                            filter_weights,
                            strides=[1, 1, 1, 1],
                            padding="VALID",
                            name="conv_description")

                        conv_username = tf.nn.conv2d(
                            embed_input_list[2],
                            filter_weights,
                            strides=[1, 1, 1, 1],
                            padding="VALID",
                            name="conv_username")

                        conv_profile_location = tf.nn.conv2d(
                            embed_input_list[3],
                            filter_weights,
                            strides=[1, 1, 1, 1],
                            padding="VALID",
                            name="conv_profile_location")


                        h_content = tf.nn.relu(tf.nn.bias_add(conv_content, filter_bias), name="relu_content")
                        h_description = tf.nn.relu(tf.nn.bias_add(conv_description, filter_bias), name="relu_description")
                        h_username = tf.nn.relu(tf.nn.bias_add(conv_username, filter_bias), name="relu_username")
                        h_profile_location = tf.nn.relu(tf.nn.bias_add(conv_profile_location, filter_bias), name="relu_profile_location")
                       
                        pooled_content = tf.nn.max_pool(
                            h_content,
                            ksize=[1, config.sequence_content - filter_size + 1, 1, 1],
                            strides=[1, 1, 1, 1],
                            padding='VALID',
                            name="pool_content")
                        output_dict["content"].append(pooled_content)

                        pooled_description = tf.nn.max_pool(
                            h_description,
                            ksize=[1, config.sequence_description - filter_size + 1, 1, 1],
                            strides=[1, 1, 1, 1],
                            padding='VALID',
                            name="pool_description")
                        output_dict["description"].append(pooled_description)


                        pooled_username = tf.nn.max_pool(
                            h_username,
                            ksize=[1, config.sequence_username - filter_size + 1, 1, 1],
                            strides=[1, 1, 1, 1],
                            padding='VALID',
                            name="pool_username")
                        output_dict["username"].append(pooled_username)

                        pooled_profile_location = tf.nn.max_pool(
                            h_profile_location,
                            ksize=[1, config.sequence_profile_location - filter_size + 1, 1, 1],
                            strides=[1, 1, 1, 1],
                            padding='VALID',
                            name="pool_profile_location")
                        output_dict["profile_location"].append(pooled_profile_location)

                return output_dict

            output_dict = build_conv_layers(self.config, embed_input_list)

            def build_h_pool(outputs, filter_num, filter_sizes):
                num_filters_total = filter_num * len(filter_sizes)
                h_pool = tf.concat(outputs, 3)
                h_pool_flat = tf.reshape(h_pool, [-1, num_filters_total])
                return h_pool_flat

            content_pool_flat = build_h_pool(output_dict["content"], self.config.filter_num, self.config.filter_sizes)
            description_pool_flat = build_h_pool(output_dict["description"], self.config.filter_num, self.config.filter_sizes)
            username_pool_flat = build_h_pool(output_dict["username"], self.config.filter_num, self.config.filter_sizes)
            profile_location_pool_flat = build_h_pool(output_dict["profile_location"], self.config.filter_num, self.config.filter_sizes)

            theta = tf.concat([content_pool_flat, description_pool_flat, username_pool_flat, profile_location_pool_flat], 1)
            theta_drop = tf.nn.dropout(theta, self.keep_prob_placeholder)
            self.theta_final = tf.concat([theta_drop, theta_misc], 1)

            with tf.name_scope("Output"):
                depth_list = [self.config.num_of_tweet_langs,
                              self.config.num_of_user_langs,
                              self.config.num_of_time_zones,
                              self.config.num_of_posting_timeslots]
                num_of_text_fields = 4
                init = tf.truncated_normal([len(self.config.filter_sizes)*self.config.filter_num*num_of_text_fields + sum(depth_list), self.config.num_classes], stddev=0.1)
                W = tf.get_variable(initializer = init, name="W")
                b = tf.get_variable(initializer = tf.constant(0.1, shape=[self.config.num_classes]), name="b")
                self.predictions = tf.nn.xw_plus_b(self.theta_final, W, b, name="predictions")
                self.classification = tf.argmax(self.predictions, 1, name = "classification")

            self.build_loss_graph()
            self.build_optimizer_graph()
        self.build_summary_graph()

    def build_loss_graph(self):
        with tf.name_scope("Loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits_v2(logits = self.predictions, labels = self.labels)
            self.loss = tf.reduce_mean(losses, name = "loss")
        with tf.name_scope("Accuracy"):
            correct_predictions = tf.equal(tf.argmax(self.predictions, 1), tf.argmax(self.labels, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")

    def build_optimizer_graph(self):
        with tf.name_scope('Optimizer'):
            optimizer = tf.train.AdamOptimizer(learning_rate=self.config.learning_rate)
            gradients, variables = zip(*optimizer.compute_gradients(loss=self.loss))
            gradients, _ = tf.clip_by_global_norm(gradients, 5.0)
            self.optimizer = optimizer.apply_gradients(zip(gradients, variables), name = "optimizer")

    def build_summary_graph(self):
            with tf.name_scope('Summary'):
                train_loss_placeholder = tf.placeholder(tf.float32, name="train_loss_placeholder")
                train_acc_placeholder = tf.placeholder(tf.float32, name="train_acc_placeholder")
                valid_loss_placeholder = tf.placeholder(tf.float32, name="valid_loss_placeholder")
                valid_acc_placeholder = tf.placeholder(tf.float32, name="valid_acc_placeholder")
                self.summary_placeholder = [train_loss_placeholder,train_acc_placeholder,valid_loss_placeholder,valid_acc_placeholder] 
                #f.summary.histogram("weights",self.weights)
                self.summary = []
                self.summary.append(tf.summary.scalar(name='train_loss', tensor=train_loss_placeholder))
                self.summary.append(tf.summary.scalar(name='train_accuracy', tensor=train_acc_placeholder))
                self.summary.append(tf.summary.scalar(name='valid_loss', tensor=valid_loss_placeholder))
                self.summary.append(tf.summary.scalar(name='valid_accuracy', tensor=valid_acc_placeholder))

    def run_epoch(self, sess, batch_size, x_text, x_misc, y, tensors, dropout, train_flag = True):

        combined_list = list(zip(x_text,x_misc, y))
        random.shuffle(combined_list)
        x_text, x_misc, y = zip(*combined_list)
        iter_num = len(y) // batch_size
        remaining = len(y) - iter_num*batch_size

        start_index = 0
        avg_loss = 0
        avg_acc = 0
        sequence_list = [self.config.sequence_content,
                         self.config.sequence_description,
                         self.config.sequence_username,
                         self.config.sequence_profile_location]
        depth_list = [self.config.num_of_tweet_langs,
                           self.config.num_of_user_langs,
                           self.config.num_of_time_zones,
                           self.config.num_of_posting_timeslots,
                           self.config.num_classes]

        def prepare_batch():
            batch_inputs_text = x_text[start_index:start_index+batch_size]
            batch_inputs_misc = x_misc[start_index:start_index+batch_size]
            batch_inputs_text = np.array(batch_inputs_text, dtype = np.int32)
            batch_inputs_misc = np.array(batch_inputs_misc, dtype = np.int32).T
            batch_labels = y[start_index:start_index+batch_size]
            batch_input_content = batch_inputs_text[:,0:sequence_list[0]]
            batch_input_desc = batch_inputs_text[:,sequence_list[0]:sum(sequence_list[0:2])]
            batch_input_username = batch_inputs_text[:,sum(sequence_list[0:2]):sum(sequence_list[0:3])]
            batch_input_profile_location = batch_inputs_text[:,sum(sequence_list[0:3]):sum(sequence_list)]
            return [batch_input_content, batch_input_desc, batch_input_username, batch_input_profile_location ,batch_inputs_misc], batch_labels

        for i in range(iter_num):
            batch_inputs, batch_labels = prepare_batch()
            feed_dict = {
                        self.input_content_placeholder : batch_inputs[0],
                        self.input_description_placeholder : batch_inputs[1],
                        self.input_username_placeholder : batch_inputs[2],
                        self.input_profile_location_placeholder : batch_inputs[3],
                        self.input_misc_placeholder : batch_inputs[4],
                        self.input_depth_placeholder : depth_list,
                        self.labels_placeholder : batch_labels,
                        self.keep_prob_placeholder : dropout
                        }
            if train_flag:
                _, loss_val, acc_val = sess.run(fetches = tensors, feed_dict = feed_dict)
            else:
                loss_val, acc_val = sess.run(fetches = tensors, feed_dict = feed_dict)
            start_index += batch_size
            avg_loss += loss_val
            avg_acc += acc_val
        if remaining > 0:
            batch_inputs, batch_labels = prepare_batch()
            feed_dict = {
                        self.input_content_placeholder : batch_inputs[0],
                        self.input_description_placeholder : batch_inputs[1],
                        self.input_username_placeholder : batch_inputs[2],
                        self.input_profile_location_placeholder : batch_inputs[3],
                        self.input_misc_placeholder : batch_inputs[4],
                        self.input_depth_placeholder : depth_list,
                        self.labels_placeholder : batch_labels,
                        self.keep_prob_placeholder : dropout
                        }
            if train_flag:
                _, loss_val, acc_val = sess.run(fetches = tensors, feed_dict = feed_dict)
            else:
                loss_val, acc_val = sess.run(fetches = tensors, feed_dict = feed_dict)

            avg_loss = (avg_loss*batch_size+loss_val*remaining)/(batch_size*iter_num+remaining)
            avg_acc = (avg_acc*batch_size+acc_val*remaining)/(batch_size*iter_num+remaining)
        else:
            avg_loss /= iter_num
            avg_acc /= iter_num
        return avg_loss, avg_acc*100


def update_summary(sess, sum_op, placeholder, value, ep, writer):
    summary = sess.run(sum_op, {placeholder:value})
    writer.add_summary(summary, ep)
    writer.flush()



def train(data_handler, cnn_model, sess, writer, train_saver, valid_saver, model_train_path, model_valid_path):
    metric_tensors = [cnn_model.loss, cnn_model.accuracy]
    dropout_train = cnn_model.config.dropout
    batch_size = cnn_model.config.batch_size
    x_train = [data_handler.ids_text["train"],data_handler.ids_misc["train"]]
    y_train = data_handler.labels["train"]
    x_valid = [data_handler.ids_text["valid"],data_handler.ids_misc["valid"]]
    y_valid = data_handler.labels["valid"]
    x_test = [data_handler.ids_text["test"],data_handler.ids_misc["test"]]
    y_test = data_handler.labels["test"]
    valid_diff = 0.025

    min_valid_loss = sess.run(cnn_model.min_valid_loss_to_save)
    print("current min valid loss: ", min_valid_loss)
    print('Starting training!')
    for ep in range(cnn_model.config.epochs_num):
        step = sess.run(cnn_model.global_step)
        print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        train_avg_loss, train_avg_acc = cnn_model.run_epoch(sess, batch_size, x_train[0], x_train[1], y_train, [cnn_model.optimizer]+metric_tensors, dropout_train)
        print("ep: ",step,' train avg loss: ', train_avg_loss, 'train avg acc: ', train_avg_acc)
        valid_avg_loss, valid_avg_acc = cnn_model.run_epoch(sess, batch_size,  x_valid[0], x_valid[1], y_valid, metric_tensors, 1, False)
        print("ep: ",step,' valid avg loss: ', valid_avg_loss, 'valid avg acc: ', valid_avg_acc)
        test_avg_loss, test_avg_acc = cnn_model.run_epoch(sess, batch_size, x_test[0], x_test[1], y_test, metric_tensors, 1, False)
        print("ep: ",step,' test avg loss: ', test_avg_loss, 'test avg acc: ', test_avg_acc)
        
        update_summary(sess, cnn_model.summary[0], cnn_model.summary_placeholder[0], train_avg_loss, step, writer)
        update_summary(sess, cnn_model.summary[1], cnn_model.summary_placeholder[1], train_avg_acc, step, writer)
        update_summary(sess, cnn_model.summary[2], cnn_model.summary_placeholder[2], valid_avg_loss, step, writer)
        update_summary(sess, cnn_model.summary[3], cnn_model.summary_placeholder[3], valid_avg_acc, step, writer)

        #increment before saving to start from the new epoch
        step = sess.run(cnn_model.increment_global_step_op)

        if (step - 1) % 30 == 0:
            valid_diff /= 2
        if min_valid_loss - valid_avg_loss > valid_diff :
            min_valid_loss = valid_avg_loss
            sess.run(cnn_model.valid_assign_op, feed_dict = {cnn_model.min_valid_loss_placeholder: valid_avg_loss})
            valid_saver.save(sess, model_valid_path + "model.ckpt", global_step = step - 1)
            print("saved valid model")

        if (step - 1) % 2 == 0:
            train_saver.save(sess, model_train_path + "model.ckpt", global_step = step - 1)
            print("saved train model")

    test_avg_loss, test_avg_acc = cnn_model.run_epoch(sess, batch_size,  x_test[0], x_test[1], y_test, metric_tensors, 1, False)
    print("test avg loss: ", test_avg_loss, 'test avg acc: ', test_avg_acc)


def test(data_handler, cnn_model, sess):
    metric_tensors = [cnn_model.loss, cnn_model.accuracy]
    x_test = [data_handler.ids_text["test"],data_handler.ids_misc["test"]]
    y_test = data_handler.labels["test"]
    test_avg_loss, test_avg_acc = cnn_model.run_epoch(sess, cnn_model.config.batch_size, x_test[0], x_test[1], y_test, metric_tensors, 1, False)
    print("test avg loss: ", test_avg_loss, 'test avg acc: ', test_avg_acc)



def save_prediction_graph(sess, cnn_model, export_path_base, model_version):
    export_path = export_path_base + str(model_version)
    print('Exporting trained model to', export_path)
    builder = tf.saved_model.builder.SavedModelBuilder(export_path)

    tensor_info_content = tf.saved_model.utils.build_tensor_info(cnn_model.input_content_placeholder)
    tensor_info_description = tf.saved_model.utils.build_tensor_info(cnn_model.input_description_placeholder)
    tensor_info_username = tf.saved_model.utils.build_tensor_info(cnn_model.input_username_placeholder)
    tensor_info_profile = tf.saved_model.utils.build_tensor_info(cnn_model.input_profile_location_placeholder)
    tensor_info_misc = tf.saved_model.utils.build_tensor_info(cnn_model.input_misc_placeholder)
    tensor_info_depth = tf.saved_model.utils.build_tensor_info(cnn_model.input_depth_placeholder)
    tensor_info_dropout = tf.saved_model.utils.build_tensor_info(cnn_model.keep_prob_placeholder)
    tensor_info_y = tf.saved_model.utils.build_tensor_info(cnn_model.classification)


    prediction_signature = (
        tf.saved_model.signature_def_utils.build_signature_def(
            inputs={'tweets_content': tensor_info_content,
                    'tweets_description': tensor_info_description,
                    'tweets_username': tensor_info_username,
                    'tweets_profile': tensor_info_profile,
                    'tweets_misc': tensor_info_misc,
                    'tweets_depth': tensor_info_depth,
                    'dropout' : tensor_info_dropout},
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
