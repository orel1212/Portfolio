import tensorflow as tf
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2
import numpy as np
from grpc.beta import implementations
import regex as re  # install by typing pip3 install regex


class SentimentAnalysis:
    def __init__(self, config):
        self.word_to_index = np.load(config.word_to_index_sentiment_path).item()
        self.timesteps = config.timesteps

    def create_ids_matrix(self, tweets):
        ids = np.zeros([len(tweets), self.timesteps], dtype=np.int32)
        unk = 'UNK'
        for i, tweet in enumerate(tweets):
            for j, word in enumerate(tweet.content_sentiment.split()):
                if word not in self.word_to_index:
                    word_tmp = word.replace("\'", "")
                    word_tmp = word_tmp.replace("!", "")
                    word_tmp = word_tmp.replace("?", "")
                    word_tmp = word_tmp.replace(",", "")
                    if word_tmp in self.word_to_index:
                        ids[i][j] = self.word_to_index[word_tmp]
                    else:
                        word_tmp = word_tmp.replace(".", "")
                        if word_tmp in self.word_to_index:
                            ids[i][j] = self.word_to_index[word_tmp]
                        else:
                            ids[i][j] = self.word_to_index[unk]
                else:
                    ids[i][j] = self.word_to_index[word]
        return ids

    def get_predictions(self, host, port, ids_matrix):
        channel = implementations.insecure_channel(host, int(port))
        stub = prediction_service_pb2.beta_create_PredictionService_stub(channel)

        # initialize a request
        request = predict_pb2.PredictRequest()
        request.model_spec.name = 'sentiment_model'
        request.model_spec.signature_name = 'prediction'

        request.inputs['tweets'].CopyFrom(
            tf.contrib.util.make_tensor_proto(ids_matrix, shape=ids_matrix.shape))
        dropout = 1.0
        request.inputs['dropout'].CopyFrom(
            tf.contrib.util.make_tensor_proto(dropout))
        result = stub.Predict(request, 120)
        predictions = result.outputs['scores'].int64_val
        return predictions

    def preprocess_tweets(self, tweets):
        for tweet in tweets:
            self.sentiment_preprocess(tweet)

    def sentiment_preprocess(self, tweet):
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
            # text = re_sub(r"/"," / ")
            text = re_sub(r"<3", "<heart>")
            text = re_sub(r"[-+]?[.\d]*[\d]+[:,.\d]*", "<number>")
            text = re_sub(r"#\S+", hashtag)
            text = re_sub(r"([!?.]){2,}", r"\1 <repeat>")
            text = re_sub(r"\b(\S*?)(.)\2{2,}\b", r"\1\2 <elong>")
            text = re_sub(r"([A-Z]){2,}", allcaps)
            return text.lower()

        text = tokenize(tweet.content_sentiment)
        text = text.split()
        text = text[:self.timesteps]
        text = " ".join(text)
        tweet.content_sentiment = text

    def predict_tweets_sentiment(self, tweets, host, port):
        self.preprocess_tweets(tweets)
        ids_matrix = self.create_ids_matrix(tweets)
        predictions = self.get_predictions(host, port, ids_matrix)
        for i, tweet in enumerate(tweets):
            tweet.set_sentiment(predictions[i])
