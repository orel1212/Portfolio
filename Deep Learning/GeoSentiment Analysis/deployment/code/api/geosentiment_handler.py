from api.config import Config
from api.sentiment_analysis import SentimentAnalysis
from api.geolocation_prediction import GeolocationPrediction

class GeoSentimentHandler:
    def __init__(self):
        print("geosentiment handler init")
        self.config = Config()
        self.sentiment_analysis = SentimentAnalysis(self.config)
        self.geolocation_prediction = GeolocationPrediction(self.config)

    def compute_predictions(self, tweets):
        self.sentiment_analysis.predict_tweets_sentiment(tweets, self.config.host, self.config.port)
        self.geolocation_prediction.predict_tweets_geolocation(tweets, self.config.host, self.config.port)

