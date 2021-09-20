import tweepy
from api.tweet import Tweet
from api.exceptions import TweepyException, ForbiddenInput

class TwitterHandler:
    def __init__(self):
        self.consumer_key = "Xac6hoSjg6ndNq6C9LZTaa5jR"
        self.consumer_secret = "LfEmjICkY5g6hlinSe1PDmGHs1wAaQme9Fadwnj1XQp1lbge1P"
        self.auth = tweepy.OAuthHandler(self.consumer_key, self.consumer_secret)


    def get_authorization_url(self):
        return self.auth.get_authorization_url()

    def get_request_token(self):
        return self.auth.request_token

    def verify_account(self, request_token, verifier_code):
        self.auth.request_token = request_token
        try:
            self.auth.get_access_token(verifier_code)
            self.api = tweepy.API(self.auth)
        except tweepy.TweepError as e:
            if '401' in e.reason:
                raise TweepyException(10, 'Reverse auth credentials error', 401)
            raise e


    def get_access_tokens(self):
        return (self.auth.access_token, self.auth.access_token_secret)

    def set_access_tokens(self, access_token, access_secret):
        self.access_token = access_token
        self.access_secret = access_secret
        self.auth.set_access_token(self.access_token, self.access_secret)
        self.api = tweepy.API(self.auth)

    def get_tweets(self, text, tweets_per_call = 100, max_tweets = 100):
        tweets = []
        filter_retweets = " -filter:retweets"
        text = "\"" + text + "\"" + filter_retweets
        try:
            cursor = tweepy.Cursor(self.api.search, q=text, count = tweets_per_call, tweet_mode='extended' )
            for tweet in cursor.items(limit = max_tweets):
                    tweets.append(Tweet(tweet._json))
            return tweets
        except tweepy.TweepError as e:
            if e.response.status_code == 429:
                if len(tweets) != 0:
                    return tweets
                raise TweepyException(8, 'Search Rate Limit', 429)
            if e.response.status_code == 401:
                raise TweepyException(9, 'Invalid or expired twitter access tokens', 401)
            if e.response.status_code == 403:
                raise ForbiddenInput()      
            raise e

    def get_trending_hashtags(self):
        try:
            trends = self.api.trends_place(1)
            data = trends[0]
            trends = data["trends"]
            trend_names = [trend["name"] for trend in trends if trend["name"][0] =="#"]
            return trend_names

        except tweepy.TweepError as e:
            if e.response.status_code == 429:
                raise TweepyException(8, 'Search Rate Limit', 429)
            if e.response.status_code == 401:
                raise TweepyException(9, 'Invalid or expired twitter access tokens', 401)
            raise e

    def get_user_id(self):
        return self.api.me().id
