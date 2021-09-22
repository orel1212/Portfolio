class ConfigLSTM:
    """Handles the hyperparameters and more of the LSTM"""
    def __init__(self, timesteps=30,
                 batch_size=1024,
                 classes_num=2,
                 epochs_num=20,
                 embed_dim=200, 
                 hidden_units=64,
                 learning_rate=0.01,
                 layers=3, 
                 beta=0,
                 dropout=1.0,
                 dropout_seed = 42
                 ):
        self.timesteps = timesteps
        self.batch_size = batch_size
        self.classes_num = classes_num
        self.epochs_num = epochs_num
        self.hidden_units = hidden_units
        self.learning_rate = learning_rate
        self.layers = layers
        self.beta = beta
        self.dropout = dropout
        self.dropout_seed = dropout_seed
        self.embed_dim = embed_dim
        



class ConfigUtil:
    """Handles the parameters of datasets path, saved models path and logging path"""

    def __init__(self, 
        dataset_name='sentiment140/',
        train_file_name='train_set',
        valid_file_name='valid_set',
        test_file_name='test_set',
        word_embeddings_name='GloVe/',
        words_list_name='words_list.npy',
        word_to_index_name='word_to_index.npy',
        word_vectors_name='word_vectors.npy',
        ids_matrix_train_name='ids_matrix_train.npy',
        ids_matrix_valid_name='ids_matrix_valid.npy',
        ids_matrix_test_name='ids_matrix_test.npy',
        word_embeddings_file_name='glove.twitter.27B.200d.txt',
        saver_path='./../saved_models/',
        export_path_base = "../serving/"):

        self.logdir = '../tensorboard_logs/'
        self.dataset_folder = '../datasets/'
        self.word_embeddings_path = '../word_embeddings/'
        self.train_models_path = '../train_models/'
        self.valid_models_path = '../valid_models/'
        self.dataset_path = self.dataset_folder + dataset_name
        self.train_path = self.dataset_path + train_file_name
        self.valid_path = self.dataset_path + valid_file_name
        self.test_path = self.dataset_path + test_file_name
        self.words_list_path = self.word_embeddings_path + word_embeddings_name + dataset_name + words_list_name
        self.word_vectors_path = self.word_embeddings_path + word_embeddings_name + dataset_name + word_vectors_name
        self.word_to_index_path = self.word_embeddings_path + word_embeddings_name + dataset_name + word_to_index_name
        self.ids_matrix_train_path = self.word_embeddings_path + word_embeddings_name + dataset_name + ids_matrix_train_name
        self.ids_matrix_valid_path = self.word_embeddings_path + word_embeddings_name + dataset_name + ids_matrix_valid_name
        self.ids_matrix_test_path = self.word_embeddings_path + word_embeddings_name + dataset_name + ids_matrix_test_name
        self.word_embed_path = self.word_embeddings_path + word_embeddings_name + word_embeddings_file_name 

        self.saver_path = saver_path
        self.export_path_base = export_path_base
