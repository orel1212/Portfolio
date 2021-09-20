import config
import numpy as np
import datetime
import string



class DataHandler:
    """A general class that handles the data.

    Parses the data, creates a vocab, word_embeddings, words_list and ids_matrix"""
    def __init__(self, datasets, config_util):
        self.config_util = config_util
        self.train_set = datasets["train_set"]
        self.valid_set = datasets["valid_set"]
        self.test_set = datasets["test_set"]
        self.ids_matrix = {"train_matrix": None,
                           "valid_matrix": None,
                           "test_matrix": None}
        self.labels = {"train" : None,
                       "valid" : None,
                       "test" : None}
        self.unknown = 'UNK'
        self.word_to_index = {}

        self.labels["train"] = [x[0] for x in datasets["train_set"]]
        self.labels["valid"] = [x[0] for x in datasets["valid_set"]]
        self.labels["test"] = [x[0] for x in datasets["test_set"]]


    def create_and_save_word_vectors(self,embed_dim):
        self.word_vectors = self.load_GloVe_model(embed_dim)
        np.save(self.config_util.word_vectors_path, self.word_vectors)
        print("Finished saving words vectors")

    def load_word_vectors(self):
        self.word_vectors = np.load(self.config_util.word_vectors_path)

    def save_word_to_index(self):
        np.save(self.config_util.word_to_index_path, self.word_to_index)
        print("Finished saving word_to_index")

    def load_word_to_index(self):
        self.word_to_index = np.load(self.config_util.word_to_index_path).item()

    def create_and_save_ids_matrix(self, timesteps):
        print("Creating ids_matrix")
        self.ids_matrix["train_matrix"] = self.create_ids_matrix(self.train_set, timesteps)
        self.ids_matrix["valid_matrix"] = self.create_ids_matrix(self.valid_set, timesteps)
        self.ids_matrix["test_matrix"] = self.create_ids_matrix(self.test_set, timesteps)
        np.save(self.config_util.ids_matrix_train_path,self.ids_matrix["train_matrix"])
        np.save(self.config_util.ids_matrix_valid_path,self.ids_matrix["valid_matrix"])
        np.save(self.config_util.ids_matrix_test_path,self.ids_matrix["test_matrix"])
        print("Finished saving ids_matrix")

    def load_ids_matrix(self):
        self.ids_matrix["train_matrix"] = np.load(self.config_util.ids_matrix_train_path)
        self.ids_matrix["valid_matrix"] = np.load(self.config_util.ids_matrix_valid_path)
        self.ids_matrix["test_matrix"] = np.load(self.config_util.ids_matrix_test_path)


    def create_ids_matrix(self, dataset, timesteps):
        ids = np.zeros([len(dataset), timesteps], dtype=np.int32)
        unk = self.unknown
        i = 0
        step = 0
        for sentence in dataset: #[0] is label, [1] is the tweet
            j = 0
            if step % 10000 == 0:
                print('ids completed: ', step)
            for word in sentence[1].split():
                if word not in self.word_to_index:
                    word_tmp = word.replace("\'","")
                    word_tmp = word_tmp.replace("!","")
                    word_tmp = word_tmp.replace("?","")
                    word_tmp = word_tmp.replace(",","")
                    if word_tmp in self.word_to_index:
                        self.word_to_index[word] = self.word_to_index[word_tmp]
                        ids[i][j] = self.word_to_index[word_tmp]
                    else:
                        word_tmp = word_tmp.replace(".","")
                        if word_tmp in self.word_to_index:
                            self.word_to_index[word] = self.word_to_index[word_tmp]
                            ids[i][j] = self.word_to_index[word_tmp]
                        else:
                            self.word_to_index[word] = self.word_to_index[unk]
                else:
                    ids[i][j] = self.word_to_index[word]
                j += 1
            i += 1
            step += 1
        return ids


    def load_GloVe_model(self, embed_dim):
        i = 0
        embd = []
        embed_unk = np.zeros([embed_dim], dtype=np.float32)
        self.word_to_index[self.unknown] = i
        embd.append(embed_unk)
        i += 1
        f = open(self.config_util.word_embed_path,'r')
        for line in f:
            splitLine = line.split()
            word = splitLine[0]
            embedding = np.array([float(val) for val in splitLine[1:]], dtype=np.float32)
            if embedding.shape == (embed_dim,):
                self.word_to_index[word] = i
                embd.append(embedding)
                i += 1
        print('Loaded GloVe!')
        f.close()
        embd = np.array(embd)
        return embd





