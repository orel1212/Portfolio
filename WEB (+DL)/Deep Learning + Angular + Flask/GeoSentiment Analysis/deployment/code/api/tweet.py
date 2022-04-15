class Tweet:

    def __init__(self, tweet_json, is_demo = False):
        self.is_demo = is_demo
        self.tid = tweet_json["id_str"]
        self.country_prediction = None
        self.sentiment = None
        self.content_geolocation = tweet_json["full_text"]
        self.content_sentiment = tweet_json["full_text"]
        self.content = tweet_json["full_text"]
        self.user_desc = tweet_json["user"]["description"]
        self.username = tweet_json["user"]["screen_name"]
        self.user_profile_location = tweet_json["user"]["location"]
        self.tweet_lang = tweet_json["lang"]
        self.user_lang = tweet_json["user"]["lang"]
        self.posting_time = tweet_json["created_at"]
        self.time_zone = tweet_json["user"]["time_zone"]

        if tweet_json["place"] is not None and tweet_json["place"]["country_code"] != "" :
            self.has_location = True
            self.country_prediction = tweet_json["place"]["country_code"]
        else:
            self.has_location = False

        self.tweet_lang_index = 0
        self.user_lang_index = 0
        self.time_zone_index = 0
        self.posting_time_slot = 0

        if self.is_demo:
            self.hashtags_set = set()
            hashtags = tweet_json['entities']['hashtags']
            for hashtag in hashtags:
                self.hashtags_set.add('#' + hashtag['text'].lower())

    def set_sentiment(self, sentiment):
        if sentiment == 0:
            self.sentiment = 'positive'
        else:
            self.sentiment = 'negative'

    def set_country(self, country):
        self.country_prediction = country


    def __str__(self):
        msg = "tid: " + self.tid + "\n"
        msg += "content_sentiment: " + self.content_sentiment + "\n"
        msg += "content_geolocation: " + self.content_geolocation + "\n"
        msg += "user_desc: " + self.user_desc + "\n"
        msg += "username: " + self.username + "\n"
        msg += "user_profile_location: " + self.user_profile_location + "\n"
        msg += "tweet_lang: " + self.tweet_lang + "\n"
        msg += "user_lang: " + self.user_lang + "\n"
        msg += "posting_time: " + self.posting_time + "\n"
        if self.time_zone != None:
            msg += "time_zone: " + self.time_zone
        else:
            msg += "time_zone: None"

        return msg

    def serialize(self):
        return {
                'text' : self.content,
                'sentiment' : self.sentiment,
                'country_prediction' : self.country_prediction,
                'created_at' : self.posting_time,
                'tweet_lang' : self.tweet_lang
        }


