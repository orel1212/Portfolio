import numpy as np
from gensim.models import KeyedVectors
from config import ConfigUtil, ConfigCNN


class Vocab:

    def __init__(self, threshold=10):
        self.filtered_word_list = []
        self.word_list = []
        self.word_freq = {}
        self.seen = set()
        self.word_to_index = {}
        self.padding = "<PAD>"
        self.add_word(self.padding)
        self.threshold = threshold
        self.word_freq[self.padding] = self.threshold

    def add_word(self, word):
        if word not in self.seen:
            self.word_list.append(word)
            self.seen.add(word)
            self.word_freq[word] = 1
        else:
            self.word_freq[word] += 1

    def filter_by_freq(self):
        self.filtered_word_list = list(filter(lambda x: self.word_freq[x] >= self.threshold, self.word_list))


class DataHandler:
    def __init__(self, datasets, config_util):
        self.vocab = Vocab()
        self.config_util = config_util
        self.train_set = datasets["train_set"]
        self.valid_set = datasets["valid_set"]
        self.test_set = datasets["test_set"]
        self.ids_text = {"train": None,
                         "valid": None,
                         "test": None}
        self.labels = {"train": None,
                       "valid": None,
                       "test": None}
        self.ids_misc = {"train": None,
                         "valid": None,
                         "test": None}

        self.labels["train"] = [x[0] for x in datasets["train_set"]]
        self.labels["valid"] = [x[0] for x in datasets["valid_set"]]
        self.labels["test"] = [x[0] for x in datasets["test_set"]]

        self.ids_misc["train"] = [x[-4:] for x in datasets["train_set"]]
        self.ids_misc["valid"] = [x[-4:] for x in datasets["valid_set"]]
        self.ids_misc["test"] = [x[-4:] for x in datasets["test_set"]]

    def create_filtered_word_list(self):
        def add_dataset_to_word_list(vocab, dataset):
            for tweet in dataset:
                for text_field in tweet[1:5]:
                    for word in text_field.split():
                        vocab.add_word(word)

        add_dataset_to_word_list(self.vocab, self.train_set)
        add_dataset_to_word_list(self.vocab, self.valid_set)
        add_dataset_to_word_list(self.vocab, self.test_set)
        self.vocab.filter_by_freq()

    def loadGloveModel(self, gloveFile):
        print("Loading Glove Model")
        f = open(gloveFile, 'r')
        model = {}
        for line in f:
            splitLine = line.split()
            word = splitLine[0]
            embedding = np.array([float(val) for val in splitLine[1:]])
            model[word] = embedding
        print("Done.", len(model), " words loaded!")
        return model

    def create_and_save_word_vectors(self, embed_dim):
        self.create_filtered_word_list()
        word_list = self.vocab.filtered_word_list
        self.word_vectors = np.zeros([len(word_list), embed_dim], dtype=np.float32)
        padding = np.zeros([embed_dim], dtype=np.float32)
        model = KeyedVectors.load_word2vec_format(self.config_util.word_embed_path, binary=True)
        # model = self.loadGloveModel(self.config_util.word_embed_path)
        print('opened w2v')
        self.word_vectors[0] = padding
        self.vocab.word_to_index[self.vocab.padding] = 0
        for i in range(1, len(word_list)):
            word = word_list[i]
            if word in model:
                self.word_vectors[i] = model[word]
            else:
                # random_vector = np.random.rand(embed_dim)
                random_vector = np.random.uniform(-1, 1, embed_dim)
                self.word_vectors[i] = random_vector
            self.vocab.word_to_index[word] = i

        np.save(self.config_util.word_vectors_path, self.word_vectors)

    def load_word_vectors(self):
        self.word_vectors = np.load(self.config_util.word_vectors_path)

    def save_word_to_index(self):
        np.save(self.config_util.word_to_index_path, self.vocab.word_to_index)
        print("Finished saving word_to_index")

    def load_word_to_index(self):
        self.vocab.word_to_index = np.load(self.config_util.word_to_index_path).item()

    def save_word_list(self):
        np.save(self.config_util.word_list_path, self.vocab.filtered_word_list)
        print("Finished saving word list")

    def load_word_list(self):
        self.vocab.filtered_word_list = np.load(self.config_util.word_list_path)

    def create_and_save_ids_text(self, sequence_list):
        print("Creating ids_text")
        self.ids_text["train"] = self.create_ids_text(self.train_set, sequence_list)
        self.ids_text["valid"] = self.create_ids_text(self.valid_set, sequence_list)
        self.ids_text["test"] = self.create_ids_text(self.test_set, sequence_list)
        np.save(self.config_util.ids_text_train_path, self.ids_text["train"])
        np.save(self.config_util.ids_text_valid_path, self.ids_text["valid"])
        np.save(self.config_util.ids_text_test_path, self.ids_text["test"])
        print("Finished saving ids_text")

    def load_ids_text(self):
        self.ids_text["train"] = np.load(self.config_util.ids_text_train_path)
        self.ids_text["valid"] = np.load(self.config_util.ids_text_valid_path)
        self.ids_text["test"] = np.load(self.config_util.ids_text_test_path)

    def create_ids_text(self, dataset, sequence_list):
        ids = np.zeros([len(dataset), sum(sequence_list)], dtype=np.int32)
        step = 0
        for i, tweet in enumerate(dataset):
            j = 0
            if step % 100000 == 0:
                print('ids completed: ', step)

            for word in tweet[1].split():
                if word in self.vocab.word_to_index:
                    ids[i][j] = self.vocab.word_to_index[word]
                j += 1

            j = sequence_list[0]
            for word in tweet[2].split():
                if word in self.vocab.word_to_index:
                    ids[i][j] = self.vocab.word_to_index[word]
                j += 1

            j = sequence_list[0] + sequence_list[1]
            for word in tweet[3].split():
                if word in self.vocab.word_to_index:
                    ids[i][j] = self.vocab.word_to_index[word]
                j += 1

            j = sequence_list[0] + sequence_list[1] + sequence_list[2]
            for word in tweet[4].split():
                if word in self.vocab.word_to_index:
                    ids[i][j] = self.vocab.word_to_index[word]
                j += 1
            step += 1
        return ids
