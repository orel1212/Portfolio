import tensorflow as tf
import numpy as np
import json
from nltk.tokenize import TweetTokenizer  # sudo pip install -U nltk
import time  # used for calculating the timesplot of posting time
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2
from grpc.beta import implementations


def load_from_json(load_path):
    with open(load_path) as infile:
        return json.load(infile)


class GeolocationPrediction:
    def __init__(self, config):
        self.word_to_index = np.load(config.word_to_index_geolocation_path).item()
        self.sequence_list = [config.sequence_content,
                              config.sequence_description,
                              config.sequence_username,
                              config.sequence_profile_location]
        self.country_list = load_from_json(config.country_list_path)
        self.time_zone_list = load_from_json(config.time_zone_list_path)
        self.tweet_lang_list = load_from_json(config.tweet_lang_list_path)
        self.user_lang_list = load_from_json(config.user_lang_list_path)
        self.depth_list = [config.num_of_tweet_langs,
                           config.num_of_user_langs,
                           config.num_of_time_zones,
                           config.num_of_posting_timeslots,
                           config.num_of_classes]

    def create_ids_text(self, tweets):
        ids = np.zeros([len(tweets), sum(self.sequence_list)], dtype=np.int32)
        for i, tweet in enumerate(tweets):
            index = 0
            for word in tweet.content_geolocation.split():
                if word in self.word_to_index:
                    ids[i][index] = self.word_to_index[word]
                index += 1

            index = self.sequence_list[0]
            for word in tweet.user_desc.split():
                if word in self.word_to_index:
                    ids[i][index] = self.word_to_index[word]
                index += 1

            index = self.sequence_list[0] + self.sequence_list[1]
            for word in tweet.username.split():
                if word in self.word_to_index:
                    ids[i][index] = self.word_to_index[word]
                index += 1

            index = self.sequence_list[0] + self.sequence_list[1] + self.sequence_list[2]
            for word in tweet.user_profile_location.split():
                if word in self.word_to_index:
                    ids[i][index] = self.word_to_index[word]
                index += 1
        return ids

    def create_misc(self, tweets):
        ids_misc = []
        for tweet in tweets:
            ids = [tweet.tweet_lang_index, tweet.user_lang_index, tweet.time_zone_index, tweet.posting_time_slot]
            ids_misc.append(ids)

        return ids_misc

    def get_predictions(self, host, port, ids_text, ids_misc):
        input_misc = np.array(ids_misc, dtype=np.int32).T
        input_content = ids_text[:, 0:self.sequence_list[0]]
        input_desc = ids_text[:, self.sequence_list[0]:sum(self.sequence_list[0:2])]
        input_username = ids_text[:, sum(self.sequence_list[0:2]):sum(self.sequence_list[0:3])]
        input_profile_location = ids_text[:, sum(self.sequence_list[0:3]):sum(self.sequence_list)]

        channel = implementations.insecure_channel(host, int(port))
        stub = prediction_service_pb2.beta_create_PredictionService_stub(channel)
        # initialize a request
        request = predict_pb2.PredictRequest()
        request.model_spec.name = 'geolocation_model'
        request.model_spec.signature_name = 'prediction'

        request.inputs['tweets_content'].CopyFrom(
            tf.contrib.util.make_tensor_proto(input_content, shape=input_content.shape))
        request.inputs['tweets_description'].CopyFrom(
            tf.contrib.util.make_tensor_proto(input_desc, shape=input_desc.shape))
        request.inputs['tweets_username'].CopyFrom(
            tf.contrib.util.make_tensor_proto(input_username, shape=input_username.shape))
        request.inputs['tweets_profile'].CopyFrom(
            tf.contrib.util.make_tensor_proto(input_profile_location, shape=input_profile_location.shape))
        request.inputs['tweets_misc'].CopyFrom(
            tf.contrib.util.make_tensor_proto(input_misc, shape=input_misc.shape))
        request.inputs['tweets_depth'].CopyFrom(
            tf.contrib.util.make_tensor_proto(self.depth_list))
        dropout = 1.0
        request.inputs['dropout'].CopyFrom(
            tf.contrib.util.make_tensor_proto(dropout))
        result = stub.Predict(request, 120)
        predictions = result.outputs['scores'].int64_val
        return predictions

    def preprocess_tweets(self, tweets):
        for i, tweet in enumerate(tweets):
            # if i == 29844:
            self.geolocation_preprocess(tweet)

    def geolocation_preprocess(self, tweet):
        def slice_helper(text, timestep):
            for i, word in enumerate(text):
                text[i] = ''.join(word.split())
            text = text[:timestep]
            text = " ".join(text)
            return text

        # check if creating a tokenizer for each tweet is expensive
        tokenizer = TweetTokenizer()
        text = tokenizer.tokenize(tweet.content_geolocation)
        tweet.content_geolocation = slice_helper(text, self.sequence_list[0])
        desc = tokenizer.tokenize(tweet.user_desc)
        tweet.user_desc = slice_helper(desc, self.sequence_list[1])
        username = tokenizer.tokenize(tweet.username)
        tweet.username = slice_helper(username, self.sequence_list[2])
        profile_location = tokenizer.tokenize(tweet.user_profile_location)
        tweet.user_profile_location = slice_helper(profile_location, self.sequence_list[3])

        if tweet.tweet_lang in self.tweet_lang_list:
            tweet.tweet_lang_index = self.tweet_lang_list.index(tweet.tweet_lang)
        else:
            tweet.tweet_lang_index = self.tweet_lang_list.index("und")

        if tweet.user_lang in self.user_lang_list:
            tweet.user_lang_index = self.user_lang_list.index(tweet.user_lang)
        else:
            tweet.user_lang_index = self.user_lang_list.index("und")

        if tweet.time_zone in self.time_zone_list:
            tweet.time_zone_index = self.time_zone_list.index(tweet.time_zone)
        else:
            tweet.time_zone_index = self.time_zone_list.index("und")

        try:
            hours = int(time.strftime('%H', time.strptime(tweet.posting_time, '%a %b %d %H:%M:%S +0000 %Y')))
            first_minute_digit = int(
                time.strftime('%M', time.strptime(tweet.posting_time, '%a %b %d %H:%M:%S +0000 %Y'))[0])
            tweet.posting_time_slot = hours * 6 + first_minute_digit
        except:
            print("Error preprocessing posting time of tweet id: ", tweet.tid)
            tweet.posting_time_slot = 0

    def predict_tweets_geolocation(self, tweets, host, port):
        tweets_without_location = [tweet for tweet in tweets if tweet.has_location == False]
        tweets_with_location = [tweet for tweet in tweets if tweet.has_location == True]
        self.preprocess_tweets(tweets_without_location)
        ids_text = self.create_ids_text(tweets_without_location)
        ids_misc = self.create_misc(tweets_without_location)
        predictions = self.get_predictions(host, port, ids_text, ids_misc)
        for i, tweet in enumerate(tweets_without_location):
            tweet.set_country(self.country_list[predictions[i]])
