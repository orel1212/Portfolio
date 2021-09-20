from grpc.beta import implementations
import numpy as np
import tensorflow as tf
from datetime import datetime
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2

import regex as re
 
def preprocess_tweet(text):

        FLAGS = re.MULTILINE | re.DOTALL

        def hashtag(text):
            text = text.group()
            hashtag_body = text[1:]
            if hashtag_body.isupper():
                result = "{}".format(hashtag_body.lower())
            result = " ".join(["<hashtag>"] + re.split(r"(?=[A-Z])", hashtag_body, flags=FLAGS))
            return result

        def allcaps(text):
            text = text.group()
            return text.lower() + " <allcaps>"


        def tokenize(text):
            # Different regex parts for smiley faces
            eyes = r"[8:=;]"
            nose = r"['`\-]?"

            # function so code less repetitive
            def re_sub(pattern, repl):
                return re.sub(pattern, repl, text, flags=FLAGS)

            text = re_sub(r"https?:\/\/\S+\b|www\.(\w+\.)+\S*", "<url>")
            text = re_sub(r"@\w+", "<user>")
            text = re_sub(r"{}{}[)dD]+|[)dD]+{}{}".format(eyes, nose, nose, eyes), "<smile>")
            text = re_sub(r"{}{}p+".format(eyes, nose), "<lolface>")
            text = re_sub(r"{}{}\(+|\)+{}{}".format(eyes, nose, nose, eyes), "<sadface>")
            text = re_sub(r"{}{}[\/|l*]".format(eyes, nose), "<neutralface>")
            #text = re_sub(r"/"," / ")
            text = re_sub(r"<3","<heart>")
            text = re_sub(r"[-+]?[.\d]*[\d]+[:,.\d]*", "<number>")
            text = re_sub(r"#\S+", hashtag)
            text = re_sub(r"([!?.]){2,}", r"\1 <repeat>")
            text = re_sub(r"\b(\S*?)(.)\2{2,}\b", r"\1\2 <elong>")
            text = re_sub(r"([A-Z]){2,}", allcaps)
            return text.lower()
        return tokenize(text)

def do_inference(ids_matrix):
    """Tests PredictionService with concurrent requests.
    Args:
    hostport: Host:port address of the Prediction Service.
    Returns:
    pred values, ground truth label
    """
    # create connection
    host, port = 'localhost', '9000'
    #host, port = 'ec2-54-91-121-153.compute-1.amazonaws.com', '9000'
    
    channel = implementations.insecure_channel(host, int(port))
    stub = prediction_service_pb2.beta_create_PredictionService_stub(channel)
   
    # initialize a request
    request = predict_pb2.PredictRequest()
    request.model_spec.name = 'example_model'
    request.model_spec.signature_name = 'prediction'
   
    request.inputs['tweets'].CopyFrom(
    tf.contrib.util.make_tensor_proto(ids_matrix, shape=ids_matrix.shape))
    dropout = 1.0
    request.inputs['dropout'].CopyFrom(
    tf.contrib.util.make_tensor_proto(dropout))

    # predict
    result = stub.Predict(request, 5.0) # 5 seconds

    return result
 
def get_tweet_prediction(tweet, word_to_index):

    ids_matrix = create_ids_matrix_without_label(tweet, 30, word_to_index)
    result = do_inference(ids_matrix)
    scores = result.outputs['scores'].int64_val
    print(scores)
    if scores[0] == 0:
        print(tweet[0],' : Positive')
    else:
        print(tweet[0],' : Negative')

def get_dataset_accruacy(ids_matrix, labels):

    result = do_inference(ids_matrix)
    #calculating accuracy
    scores = result.outputs['scores'].int64_val

    score = 0
    for i,x in enumerate(scores):
        label = [1, 0]
        if x == 1: #negative
            label = [0, 1]
        if label == labels[i]:
            score += 1

    return score*100/len(labels)

def create_ids_matrix_without_label(dataset, timesteps, word_to_index):
        ids = np.zeros([len(dataset), timesteps], dtype=np.int32)
        unk = 'UNK'
        i = 0
        step = 0
        for sentence in dataset: #[0] is label, [1] is the tweet
            j = 0
            for word in sentence[:timesteps].split(" "):
                if word not in word_to_index:
                    word_tmp = word.replace("\'","")
                    word_tmp = word_tmp.replace("!","")
                    word_tmp = word_tmp.replace("?","")
                    word_tmp = word_tmp.replace(",","")
                    if word_tmp in word_to_index:
                        word_to_index[word] = word_to_index[word_tmp]
                        ids[i][j] = word_to_index[word_tmp]
                    else:
                        word_tmp = word_tmp.replace(".","")
                        if word_tmp in word_to_index:
                            word_to_index[word] = word_to_index[word_tmp]
                            ids[i][j] = word_to_index[word_tmp]
                        else:
                            word_to_index[word] = word_to_index[unk]
                            # added this line in case word_to_index[unk] is not 0
                            ids[i][j] = word_to_index[unk]
                else:
                    ids[i][j] = word_to_index[word]
                j += 1
            i += 1
            step += 1
        return ids

def create_ids_matrix(dataset, timesteps, word_to_index):
        ids = np.zeros([len(dataset), timesteps], dtype=np.int32)
        unk = 'UNK'
        i = 0
        step = 0
        for sentence in dataset: #[0] is label, [1] is the tweet
            j = 0
            for word in sentence[1].split(" "):
                if word not in word_to_index:
                    word_tmp = word.replace("\'","")
                    word_tmp = word_tmp.replace("!","")
                    word_tmp = word_tmp.replace("?","")
                    word_tmp = word_tmp.replace(",","")
                    if word_tmp in word_to_index:
                        word_to_index[word] = word_to_index[word_tmp]
                        ids[i][j] = word_to_index[word_tmp]
                    else:
                        word_tmp = word_tmp.replace(".","")
                        if word_tmp in word_to_index:
                            word_to_index[word] = word_to_index[word_tmp]
                            ids[i][j] = word_to_index[word_tmp]
                        else:
                            word_to_index[word] = word_to_index[unk]
                            # added this line in case word_to_index[unk] is not 0
                            ids[i][j] = word_to_index[unk]
                else:
                    ids[i][j] = word_to_index[word]
                j += 1
            i += 1
            step += 1
        return ids





def main():

    ids_matrix_path = '/home/timor/project/geosentiment_analysis/sentiment_analysis/word_embeddings/GloVe/sentiment140/ids_matrix_test.npy'
    ids_matrix = np.load(ids_matrix_path)
    dataset_path =  '/home/timor/project/geosentiment_analysis/sentiment_analysis/datasets/sentiment140/test_set_preprocessed.npy'
    dataset = np.load(dataset_path)
    print(len(dataset))
    labels = [x[0] for x in dataset]
    acc = get_dataset_accruacy(ids_matrix, labels)
    print("Accuracy is: ", acc)


    word_to_index_path = '/home/timor/project/geosentiment_analysis/sentiment_analysis/word_embeddings/GloVe/sentiment140/word_to_index.npy'
    word_to_index = np.load(word_to_index_path).item()

    tweet = input("input a tweet or press 0 to exit the loop: ")
    tweet = [tweet]
    while tweet != '0':
        tweet[0] = preprocess_tweet(tweet[0])
        print(tweet[0])
        get_tweet_prediction(tweet, word_to_index)
        tweet = input("input a tweet or press 0 to exit the loop: ")
        tweet = [tweet]



if __name__ == "__main__":
    main()
