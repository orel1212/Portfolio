class ConfigCNN:
    """Handles the hyperparameters and more of the CNN"""

    def __init__(self,
                 sequence_content=30,
                 sequence_description=35,
                 sequence_username=10,
                 sequence_profile_location=8,
                 num_of_tweet_langs=42,
                 num_of_user_langs=61,
                 num_of_time_zones=320,
                 num_of_posting_timeslots=144,
                 misc_size=4,
                 filter_sizes=[3, 4, 5],
                 filter_num=128,
                 num_classes=190,
                 learning_rate=0.001,
                 embed_dim=300,
                 dropout=0.5,
                 dropout_seed=42,
                 batch_size=1024,
                 epochs_num=20):
        self.sequence_content = sequence_content
        self.sequence_description = sequence_description
        self.sequence_profile_location = sequence_profile_location
        self.sequence_username = sequence_username
        self.filter_sizes = filter_sizes
        self.filter_num = filter_num
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.embed_dim = embed_dim
        self.dropout = dropout
        self.dropout_seed = dropout_seed
        self.batch_size = batch_size
        self.misc_size = misc_size
        self.num_of_tweet_langs = num_of_tweet_langs
        self.num_of_user_langs = num_of_user_langs
        self.num_of_time_zones = num_of_time_zones
        self.num_of_posting_timeslots = num_of_posting_timeslots
        self.epochs_num = epochs_num


class ConfigUtil:
    """Handles the parameters of datasets path, saved models path and logging path"""

    def __init__(self,
                 dataset_name='wnut/',
                 train_file_name='train_set',
                 valid_file_name='valid_set',
                 test_file_name='test_set',
                 word_embeddings_name='word2vec/',
                 word_list_name='word_list.npy',
                 word_to_index_name='word_to_index.npy',
                 word_vectors_name='word_vectors.npy',
                 ids_text_train_name='ids_text_train.npy',
                 ids_text_valid_name='ids_text_valid.npy',
                 ids_text_test_name='ids_text_test.npy',
                 word_embeddings_file_name='GoogleNews-vectors-negative300.bin',
                 # word_embeddings_file_name = "glove.twitter.27B.200d.txt",
                 saver_path='../saved_models/',
                 export_path_base="../serving/",
                 country_list_name="country_list",
                 user_lang_list_name="user_lang_list",
                 tweet_lang_list_name="tweet_lang_list",
                 time_zone_list_name="time_zone_list"):
        self.logdir = '../tensorboard_logs/'
        self.dataset_folder = '../datasets/'
        self.word_embeddings_path = '../word_embeddings/'
        self.train_models_path = '../train_models/'
        self.valid_models_path = '../valid_models/'
        self.dataset_path = self.dataset_folder + dataset_name
        self.train_path = self.dataset_path + train_file_name
        self.valid_path = self.dataset_path + valid_file_name
        self.test_path = self.dataset_path + test_file_name
        self.word_list_path = self.word_embeddings_path + word_embeddings_name + dataset_name + word_list_name
        self.word_vectors_path = self.word_embeddings_path + word_embeddings_name + dataset_name + word_vectors_name
        self.word_to_index_path = self.word_embeddings_path + word_embeddings_name + dataset_name + word_to_index_name
        self.ids_text_train_path = self.word_embeddings_path + word_embeddings_name + dataset_name + ids_text_train_name
        self.ids_text_valid_path = self.word_embeddings_path + word_embeddings_name + dataset_name + ids_text_valid_name
        self.ids_text_test_path = self.word_embeddings_path + word_embeddings_name + dataset_name + ids_text_test_name
        self.word_embed_path = self.word_embeddings_path + word_embeddings_name + word_embeddings_file_name
        self.saver_path = saver_path
        self.export_path_base = export_path_base
        self.country_list_path = self.dataset_path + country_list_name
        self.time_zone_list_name = self.dataset_path + time_zone_list_name
        self.user_lang_list_name = self.dataset_path + user_lang_list_name
        self.tweet_lang_list_name = self.dataset_path + tweet_lang_list_name
