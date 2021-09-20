import numpy as np
import csv
import json
from random import shuffle


def save_to_json(save_path, data):
    with open(save_path, 'w') as outfile:
        json.dump(data, outfile)


def parse_sentiment140(filename):
    dataset=[]
    with open(filename, 'r',encoding ='latin-1') as file:
        reader = csv.reader(file, delimiter=',')
        lst = list(reader)
        for line in lst:
            sentim = line[0]
            text = line[5]
            label = []
            if sentim == '4':#positive
                label = [1, 0]
                dataset.append([label, text])
            elif sentim == '0':#negative
                label = [0, 1]
                dataset.append([label, text])
    return dataset


def save_train_valid(dataset, dataset_path, percentage):
    shuffle(dataset)
    dataset = dataset
    shuffle(dataset)
    valid_set_length = int(len(dataset) * percentage)
    train_set_length = len(dataset)  - valid_set_length
    valid_set = dataset[train_set_length:]
    train_set = dataset[:train_set_length]  
    save_to_json(dataset_path+"train_set", train_set)
    save_to_json(dataset_path+"valid_set", valid_set)   



#PARSE SENTIMENT140
dataset_name='sentiment140/'
dataset_folder = '../datasets/'
filename_train='training.1600000.processed.noemoticon.csv'
filename_test= 'testdata.manual.2009.06.14.csv'
dataset_path = dataset_folder + dataset_name
dataset = parse_sentiment140(dataset_path+filename_train)
save_train_valid(dataset,dataset_path,0.01)
save_to_json(dataset_path+"test_set",parse_sentiment140(dataset_path+filename_test))


