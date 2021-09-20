from api.tweet import Tweet
import json
from os import listdir
from os.path import splitext
from api.exceptions import ForbiddenInput
import re
class DemoHandler():
    def __init__(self, path):
        print("DemoHandler init")
        self.path = path
        self.hashtag_filenames = self.get_demo_tweets_filenames_from_folder()
        self.hashtags = ['#' + splitext(hashtag)[0].lower() for hashtag in self.hashtag_filenames]
        self.load_all_tweets()
        print("DemoHandler done init")

    def get_demo_tweets_filenames_from_folder(self):
        return [f for f in listdir(self.path) if f.endswith(".json")]

    def get_hashtags(self):
        return self.hashtags

    def load_all_tweets(self):
        self.tweets = []
        for hashtag_filename in self.hashtag_filenames:
            with open(self.path + '/' + hashtag_filename) as f:
                for line in f:
                    self.tweets.append(Tweet(json.loads(line), is_demo = True))

    def preprocess_search_input(self, search_input):
        hashtags_OR = []
        search_input = re.sub(r'\b(\w+)( \1\b)+', r'\1', search_input)
        search_input = search_input.lstrip(' ')
        search_input = search_input.rstrip(' ')
        search_input_list = search_input.split()
        if search_input_list[0] == "or":
            raise ForbiddenInput()
        if search_input_list[-1] == "or":
            raise ForbiddenInput()
        for input_element in search_input_list:
            if input_element != 'or' and input_element[0] != '#':
                raise ForbiddenInput()

        for hashtags in search_input.split(" or "):
            if len(hashtags) != 0:
                hashtags_AND_set = set(hashtags.split())
                hashtags_OR.append(hashtags_AND_set)
        return hashtags_OR

    def search_by_hashtags(self, hashtags_OR):
        found_tweets = []
        for tweet in self.tweets:
            for hashtags_set in hashtags_OR:
                if hashtags_set.issubset(tweet.hashtags_set):
                    found_tweets.append(tweet)
                    break
        return found_tweets
