import os
import sys
import unittest
from unittest.mock import patch, PropertyMock
import api
import json
from api.models import UserModel
from api import app, db
from flask_jwt_extended import (create_access_token, create_refresh_token, jwt_required, jwt_refresh_token_required, get_jwt_identity, get_raw_jwt)

def get_mock_tweets():
    tweets = []
    with open("api/tests/data/test_tweets.json") as f:
        for line in f:
            tweets.append(api.tweet.Tweet(json.loads(line), is_demo = True))
    return tweets

def mock_jwt_required(id = 1, is_refresh = False):
    token = create_access_token(identity = id)
    headers = {
        'Authorization': 'Bearer {}'.format(token)
    }
    return headers


def get_json_content_header():
    headers = {
        'Content-Type': 'application/json'
    }
    return headers 

def get_mock_config():
    config = api.config.Config() 
    config.word_to_index_sentiment_path = "api/tests/data/word_to_index_sentiment.npy"
    config.word_to_index_geolocation_path = "api/tests/data/word_to_index_geolocation.npy"
    config.country_list_path = "api/tests/data/country_list"
    config.time_zone_list_path = "api/tests/data/time_zone_list"
    config.tweet_lang_list_path = "api/tests/data/tweet_lang_list"
    config.user_lang_list_path = "api/tests/data/user_lang_list"
    return config


class BasicTests(unittest.TestCase):

    ############################
    #### setup and teardown ####
    ############################

    # executed prior to each test
    def setUp(self):
        app.config['TESTING'] = True
        app.config['WTF_CSRF_ENABLED'] = False
        app.config['DEBUG'] = False
        app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///tests/test.db'

        self.app = app.test_client()
        db.drop_all()
        db.create_all()


        self.assertEqual(app.debug, False)

    # executed after each test
    def tearDown(self):
        pass


    def add_user_to_db(self, id = 1, token = 12, secret = 123):
        test_user = UserModel(
        id = id,
        access_token = token,
        access_secret = secret
        )

        test_user.save_to_db()

###############
#### tests ####
###############

    def test_twitter_auth(self):
        response = self.app.get('/api/twitter_auth')
        data = json.loads(response.data.decode('utf8'))
        keys = ['request_token', 'request_url']
        if not all (k in data.keys() for k in keys):
            self.fail("The keys are not matching")

    def test_twitter_verify_no_json_given(self):
        headers = get_json_content_header()
        response = self.app.post('/api/twitter_verify_auth')
        data = json.loads(response.data.decode('utf8'))
        self.assertEqual(data["code"],12)
        self.assertEqual(data["message"], "No JSON given")    

    def test_twitter_verify_invalid_json(self):
        headers = get_json_content_header()
        response = self.app.post('/api/twitter_verify_auth', headers = headers, data = "{invalid: []")
        data = json.loads(response.data.decode('utf8'))
        self.assertEqual(data["code"],11)
        self.assertEqual(data["message"], "JSON Format Invalid")

    @patch.object(api.twitter_handler.TwitterHandler, "verify_account")
    @patch.object(api.twitter_handler.TwitterHandler, 'get_access_tokens')
    @patch.object(api.twitter_handler.TwitterHandler, 'get_user_id')
    def test_twitter_verify_auth(self, mock_get_user_id, mock_get_access_tokens, mock_verify_account):
        mock_verify_account.return_value = True
        mock_get_access_tokens.return_value = (12,123)
        mock_get_user_id.return_value = 1
        self.add_user_to_db(1,12,123)
        headers = get_json_content_header()
        response = self.app.post('/api/twitter_verify_auth', headers = headers, data = '{"verifier_code": "123", "request_token": "123"}')
        data = json.loads(response.data.decode('utf8'))
        keys = ['access_token', 'refresh_token']
        if not all (k in data.keys() for k in keys):
            self.fail("The keys are not matching")

    @patch.object(api.twitter_handler.TwitterHandler, "verify_account")
    @patch.object(api.twitter_handler.TwitterHandler, 'get_access_tokens')
    @patch.object(api.twitter_handler.TwitterHandler, 'get_user_id')
    def test_twitter_verify_auth_new_user(self, mock_get_user_id, mock_get_access_tokens, mock_verify_account):
        mock_verify_account.return_value = True
        mock_get_access_tokens.return_value = (12,123)
        mock_get_user_id.return_value = 2
        headers = get_json_content_header()
        response = self.app.post('/api/twitter_verify_auth', headers = headers, data = '{"verifier_code": "123", "request_token": "123"}')
        data = json.loads(response.data.decode('utf8'))
        keys = ['access_token', 'refresh_token']
        if not all (k in data.keys() for k in keys):
            self.fail("The keys are not matching")

        if UserModel.find_by_id(mock_get_user_id.return_value) is None:
            self.fail("Error adding user to db")


    def test_not_found(self):
        response = self.app.get('/not_found', follow_redirects=True)
        data = json.loads(response.data.decode('utf8'))
        self.assertEqual(data["code"],3)
        self.assertEqual(data["message"], "Path Not Found")
        self.assertEqual(response.status_code, 404)

    @patch.object(api.twitter_handler.TwitterHandler, 'get_trending_hashtags')
    def test_api_trends(self, mock_get_trending_hashtags):
        self.add_user_to_db(1,12,123)
        mock_get_trending_hashtags.return_value=['#trump', '#putin']
        with app.app_context():
            headers = mock_jwt_required()
        response = self.app.get('/api/trends', headers=headers);
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data.decode('utf8'))
        trends = ['#trump', '#putin']
        if not all (k in data["trends"] for k in trends):
            self.fail("The trends are not matching")
        

    def test_sentiment_preprocess(self):
        original_tweets = get_mock_tweets()
        original_tweet = original_tweets[0]
        original_tweet.content_sentiment = 'HELLO WORLD <3 123'
        sentiment_analysis = api.sentiment_analysis.SentimentAnalysis(api.config.Config())
        sentiment_analysis.preprocess_tweets(original_tweets)
        self.assertEqual(original_tweet.content_sentiment, "hello <allcaps> world <allcaps> <heart> <number>")

    def test_geolocation_preprocess(self):
        original_tweets = get_mock_tweets()
        original_tweet = original_tweets[0]
        original_tweet.content_geolocation = 'HELLO WORLD <3 123 #gmail!!!'
        original_tweet.tweet_lang = 'en'
        original_tweet.user_lang = 'en'
        original_tweet.posting_time = 'Tue May 15 08:50:07 +0000 2018'
        original_tweet.time_zone = 'Central Time (US & Canada)'
        geolocation_prediction = api.geolocation_prediction.GeolocationPrediction(get_mock_config())
        geolocation_prediction.preprocess_tweets(original_tweets)

        expected_tweet_lang_index = 6
        expected_user_lang_index = 6
        expected_time_zone_index = 63
        expected_posting_time_slot = 53
        expected_content_geolocation = 'HELLO WORLD <3 123 #gmail ! ! !'

        self.assertEqual(original_tweet.tweet_lang_index, expected_tweet_lang_index)
        self.assertEqual(original_tweet.user_lang_index, expected_user_lang_index)
        self.assertEqual(original_tweet.time_zone_index, expected_time_zone_index)
        self.assertEqual(original_tweet.posting_time_slot, expected_posting_time_slot)
        self.assertEqual(original_tweet.content_geolocation, expected_content_geolocation)

    def test_ids_sentiment(self):

        original_tweets = get_mock_tweets()
        original_tweet = original_tweets[0]
        original_tweet.content_sentiment = 'HELLO WORLD <3 123'

        sentiment_analysis = api.sentiment_analysis.SentimentAnalysis(get_mock_config())
        sentiment_analysis.preprocess_tweets(original_tweets)
        ids_matrix = sentiment_analysis.create_ids_matrix(original_tweets)

        fill_ids_matrix = [0] * 24
        expected_tweet_ids_matrix = [997, 19, 368, 19, 137, 8] + fill_ids_matrix
        self.assertSequenceEqual(list(ids_matrix[0]),expected_tweet_ids_matrix)

    def test_ids_text_geolocation(self):
        original_tweets = get_mock_tweets()
        original_tweet = original_tweets[0]
        original_tweet.content_geolocation = 'HELLO WORLD <3 123 #gmail!!!'
        original_tweet.user_desc = "I'm a cat"
        original_tweet.username = "username too long"
        original_tweet.user_profile_location = "no location"

        geolocation_prediction = api.geolocation_prediction.GeolocationPrediction(get_mock_config())
        geolocation_prediction.preprocess_tweets(original_tweets)

        ids_text = geolocation_prediction.create_ids_text(original_tweets)

        expcted_tweet_ids_text = [0, 3105, 1136, 1301, 0, 1, 1, 1] + 22 * [0]
        expcted_tweet_ids_text += [61, 145, 2307] + 32 * [0] 
        expcted_tweet_ids_text += [0, 250, 2219] + 7 * [0] 
        expcted_tweet_ids_text += [180, 3656] + 6 * [0]

        self.assertSequenceEqual(list(ids_text[0]),expcted_tweet_ids_text)

    def test_ids_misc_gelocation(self):
        original_tweets = get_mock_tweets()
        first_tweet = original_tweets[0]
        last_tweet = original_tweets[-1]

        first_tweet.tweet_lang_index = 0
        first_tweet.user_lang_index = 0
        first_tweet.time_zone_index = 0
        first_tweet.posting_time_slot = 0

        last_tweet.tweet_lang_index = 1
        last_tweet.user_lang_index = 1
        last_tweet.time_zone_index = 1
        last_tweet.posting_time_slot = 1

        geolocation_prediction = api.geolocation_prediction.GeolocationPrediction(get_mock_config())

        ids_misc = geolocation_prediction.create_misc(original_tweets)

        expected_tweet_ids_misc_first = [0, 0, 0, 0]
        expected_tweet_ids_misc_last = [1, 1, 1, 1]
        self.assertSequenceEqual(list(ids_misc[0]),expected_tweet_ids_misc_first)
        self.assertSequenceEqual(list(ids_misc[-1]),expected_tweet_ids_misc_last)


    @patch.object(api.demo_handler.DemoHandler, 'get_hashtags')
    def test_demo_trends(self, mock_get_hashtags):
        mock_get_hashtags.return_value=['#trump', '#putin']
        response = self.app.get('/api/demo/trends');
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data.decode('utf8'))
        trends = ['#trump', '#putin']
        if not all (k in data["trends"] for k in trends):
            self.fail("The trends are not matching")


    def test_demo_predictions_invalid_search(self):
        search_input = "or %trump"
        response = self.app.get('/api/demo/predictions/' + search_input);
        data = json.loads(response.data.decode('utf8'))
        self.assertEqual(data["code"],2)
        self.assertEqual(response.status_code, 403)

    @patch.object(api.twitter_handler.TwitterHandler, "get_tweets")
    @patch.object(api.geolocation_prediction.GeolocationPrediction, 'get_predictions')
    @patch.object(api.sentiment_analysis.SentimentAnalysis, 'get_predictions')
    def test_prediction(self, mock_get_sentiment_predictions, mock_get_geolocation_predictions, mock_get_tweets):
        self.add_user_to_db(1,123,1234)
        tweets = get_mock_tweets()
        mock_get_tweets.return_value = tweets
        mock_get_sentiment_predictions.return_value = len(tweets) * [0]
        mock_get_geolocation_predictions.return_value = len(tweets) * [0]
        with app.app_context():
            headers = mock_jwt_required()
        response = self.app.get('/api/predictions/%23trump', headers=headers)
        data = json.loads(response.data.decode('utf8'))
        first_tweet_predictions = [data['tweets'][0]['sentiment'], data['tweets'][0]['country_prediction']]
        last_tweet_predictions = [data['tweets'][-1]['sentiment'], data['tweets'][-1]['country_prediction']]
        excpected_sentiment_prediction = 'positive'
        excpected_geolocation_prediction = api.views.geosentiment_handler.geolocation_prediction.country_list[0]
        self.assertEqual(first_tweet_predictions[0], excpected_sentiment_prediction)
        self.assertEqual(last_tweet_predictions[0], excpected_sentiment_prediction)
        self.assertEqual(first_tweet_predictions[1], excpected_geolocation_prediction)
        self.assertEqual(last_tweet_predictions[1], excpected_geolocation_prediction)
        self.assertEqual(response.status_code, 200)


    @patch.object(api.demo_handler.DemoHandler, 'search_by_hashtags')
    @patch.object(api.geolocation_prediction.GeolocationPrediction, 'get_predictions')
    @patch.object(api.sentiment_analysis.SentimentAnalysis, 'get_predictions')
    def test_demo_prediction(self, mock_get_sentiment_predictions, mock_get_geolocation_predictions, mock_search_by_hashtags): 
        tweets = get_mock_tweets()
        mock_search_by_hashtags.return_value = tweets
        mock_get_sentiment_predictions.return_value = len(tweets) * [0]
        mock_get_geolocation_predictions.return_value = len(tweets) * [0]
        mocked_config = get_mock_config()
        response = self.app.get('/api/demo/predictions/%23hello %23world')
        data = json.loads(response.data.decode('utf8'))

        first_tweet_predictions = [data['tweets'][0]['sentiment'], data['tweets'][0]['country_prediction']]
        last_tweet_predictions = [data['tweets'][-1]['sentiment'], data['tweets'][-1]['country_prediction']]
        excpected_sentiment_prediction = 'positive'
        excpected_geolocation_prediction = api.views.geosentiment_handler.geolocation_prediction.country_list[0]
        self.assertEqual(first_tweet_predictions[0], excpected_sentiment_prediction)
        self.assertEqual(last_tweet_predictions[0], excpected_sentiment_prediction)
        self.assertEqual(first_tweet_predictions[1], excpected_geolocation_prediction)
        self.assertEqual(last_tweet_predictions[1], excpected_geolocation_prediction)
        self.assertEqual(response.status_code, 200)



if __name__ == "__main__":
    unittest.main()