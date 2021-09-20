import numpy as np
from preprocessor import Preprocessor
import util
import cnn
import argparse
from config import ConfigUtil
import tensorflow as tf
import json


def init_parser():
    epilog_helper_msg="if you decide to supply option, then it supposes that "
    epilog_helper_msg+="you want to create a file with the output of the action of that option."
    epilog_helper_msg+="\nOtherwise it will just load the file.\nFor example:if you specified -p, "
    epilog_helper_msg+="then it will parse the dataset \nand store the file in the disk.\nOtherwise, it will just load the file.\n"
    arg_parser = argparse.ArgumentParser(prog='main.py',
     usage='%(prog)s [options]',
     epilog=epilog_helper_msg,
     formatter_class=argparse.RawTextHelpFormatter)
    arg_parser.add_argument('-p','--preprocess', action='store_true',default=False,help='preprocessing the data set',dest="preprocess")
    arg_parser.add_argument('-we','--word_embeds', action='store_true',default=False,help='creating the embeddings of the set',dest="create_embeddings")
    arg_parser.add_argument('-im','--ids_matrix', action='store_true',default=False,help='creating the ids matrix of the set',dest="create_ids_matrix")
    arg_parser.add_argument('-lt','--load_train', action='store_true',default=False,help='loading latest saved train model to train',dest="load_train")
    arg_parser.add_argument('-lv','--load_valid', action='store_true',default=False,help='loading latest saved valid model to train',dest="load_valid")
    arg_parser.add_argument('-t','--test_model', action='store_true',default=False,help='test loaded model',dest="test_model")
    arg_parser.add_argument('-d','--directory', action='store',help='directory name of the logs and the model',dest="dir_name")
    arg_parser.add_argument('-s','--serving', action='store',help='version of the model as integer exported by tf_serving',dest="serving_version")
    return arg_parser

def reconfigure_flags(flags):
    if flags["dir_name"] == None:
        print("you must enter dir name with -d option! use -h option for help!")
        exit()
    more_than_one_load_flag=False
    if flags["load_train"] and flags["load_valid"]:
        print("cannot load more than one model at the time! use -h option for help!")
        exit()
    if flags["test_model"] and not flags["load_train"] and not flags["load_valid"]:
        print("you must specify either load valid or train model with the test option! use -h option for help!")
        exit()
    if flags["serving_version"] and not flags["load_train"] and not flags["load_valid"]:
        print("you must specify either load valid or train model with the serving option! use -h option for help!")
        exit()
    if flags["serving_version"]:
        try:
            int(flags["serving_version"])
        except:
            print("serving version must be an integer! use -h option for help")
            exit()
    create_flags_level=0
    if flags["preprocess"]:
        create_flags_level=2
    if flags["create_embeddings"]:
        create_flags_level=1
    if create_flags_level>0:
        flags["create_ids_matrix"]=True
        if create_flags_level>1:
            flags["create_embeddings"]=True
    return flags


def load_from_json(load_path):
    with open(load_path) as infile:
        return json.load(infile)

def save_to_json(save_path, data):
    with open(save_path, 'w') as outfile:
        json.dump(data, outfile)

def main():
    arg_parser=init_parser()
    flags = reconfigure_flags(vars(arg_parser.parse_args()))
    config_util = ConfigUtil()
    cnn_model = cnn.CNN()
    sequence_list = [cnn_model.config.sequence_content,
                     cnn_model.config.sequence_description,
                     cnn_model.config.sequence_username,
                     cnn_model.config.sequence_profile_location]
    datasets = {}
    datasets["train_set"] = load_from_json(config_util.train_path)
    datasets["valid_set"] = load_from_json(config_util.valid_path)
    datasets["test_set"] = load_from_json(config_util.test_path)

    if flags["preprocess"]:
        misc_categories = {}
        misc_categories["country"] = load_from_json(config_util.country_list_path)
        misc_categories["time_zone"] = load_from_json(config_util.time_zone_list_name)
        misc_categories["user_lang"] = load_from_json(config_util.user_lang_list_name)
        misc_categories["tweet_lang"] = load_from_json(config_util.tweet_lang_list_name)
        preprocessor = Preprocessor(datasets,sequence_list, misc_categories)
        save_to_json(config_util.dataset_path+"train_set_preprocessed", datasets["train_set"])
        save_to_json(config_util.dataset_path+"valid_set_preprocessed", datasets["valid_set"])
        save_to_json(config_util.dataset_path+"test_set_preprocessed", datasets["test_set"])
        print("saved preprocessed datasets")

    else:
        datasets["train_set"] = load_from_json(config_util.dataset_path+"train_set_preprocessed")
        datasets["valid_set"] = load_from_json(config_util.dataset_path+"valid_set_preprocessed")
        datasets["test_set"] = load_from_json(config_util.dataset_path+"test_set_preprocessed")
        print("Done loading datasets")

    data_handler = util.DataHandler(datasets,config_util)


    if flags["create_embeddings"]:
        data_handler.create_and_save_word_vectors(cnn_model.config.embed_dim)
        data_handler.save_word_to_index()
    else:
        #we only load the word_vectors from file if it exists and does not exist in the graph
        if not flags["load_train"] and not flags["load_valid"]:
            data_handler.load_word_vectors()

    if flags["create_ids_matrix"]:
        if not flags["create_embeddings"]:
            data_handler.load_word_to_index()
        data_handler.create_and_save_ids_text(sequence_list)
    else:
        data_handler.load_ids_text()



    dir_name = flags["dir_name"]
    model_train_path = config_util.saver_path + "train/" + dir_name + "/"
    model_valid_path = config_util.saver_path + "valid/" + dir_name + "/"
    logdir = config_util.logdir + dir_name
    sess = tf.Session()
    if not flags["load_train"] and not flags["load_valid"]:
        cnn_model.build_train_graph(data_handler.word_vectors.shape)
        train_saver = tf.train.Saver(max_to_keep = 5)
        valid_saver = tf.train.Saver(max_to_keep = 2)
        sess.run(tf.global_variables_initializer())
        sess.run(cnn_model.embedding_init, feed_dict={cnn_model.embedding_placeholder: data_handler.word_vectors})
        writer = tf.summary.FileWriter(logdir,sess.graph)
        data_handler.word_vectors = None #to free space
        train_saver.save(sess, model_train_path + "model.ckpt")
        valid_saver.save(sess, model_valid_path + "model.ckpt")
        cnn.train(data_handler,cnn_model, sess, writer, train_saver, valid_saver, model_train_path, model_valid_path)
        writer.close()
    else:
        if flags["load_train"]:
            latest_model = tf.train.latest_checkpoint(model_train_path)

        elif flags["load_valid"]:
            latest_model = tf.train.latest_checkpoint(model_valid_path)

        saver = tf.train.import_meta_graph(latest_model + '.meta')
        print("restored train model: ", latest_model)
        saver.restore(sess, latest_model)

        cnn_model.restore_train_graph(sess)
        train_saver = tf.train.Saver(max_to_keep = 5)
        valid_saver = tf.train.Saver(max_to_keep = 2)
        writer = tf.summary.FileWriter(logdir,sess.graph)

        if not flags["test_model"] and flags["serving_version"] == None:
            cnn.train(data_handler,cnn_model, sess, writer, train_saver, valid_saver, model_train_path, model_valid_path)
        else: 
            if flags["test_model"]:
                cnn.test(data_handler,cnn_model, sess)
            if flags["serving_version"] != None:
                cnn.save_prediction_graph(sess, cnn_model, config_util.export_path_base, flags["serving_version"])
        writer.close()

    sess.close()


if __name__ == "__main__":
    main()