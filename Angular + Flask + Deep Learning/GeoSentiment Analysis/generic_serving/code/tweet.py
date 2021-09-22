class Tweet:

    def __init__(self, tweet_json):
        #self.tid = tweet_json["id_str"]
        self.country = None
        self.country_prediction = None
        self.original_sentiment = None
        self.sentiment = None
        if "text" in tweet_json:
            self.content = tweet_json["text"]
        else:
            self.content = tweet_json["full_text"]
        self.content_geolocation = self.content
        self.content_sentiment = self.content

        self.user_desc = tweet_json["user"]["description"]
        self.username = tweet_json["user"]["name"]
        self.user_profile_location = tweet_json["user"]["location"]
        self.tweet_lang = tweet_json["lang"]
        self.user_lang = tweet_json["user"]["lang"]
        self.posting_time = tweet_json["created_at"]
        self.time_zone = tweet_json["user"]["time_zone"]

        if tweet_json["place"] is not None and tweet_json["place"]["country_code"] != "" :
            self.has_location = True
            self.country = tweet_json["place"]["country_code"]
        else:
            self.has_location = False

        if "sentiment" in tweet_json:
            self.original_sentiment = tweet_json["sentiment"]

        self.tweet_lang_index = 0
        self.user_lang_index = 0
        self.time_zone_index = 0
        self.posting_time_slot = 0

    def set_sentiment(self, sentiment):
        if sentiment == 0:
            self.sentiment = 'positive'
        else:
            self.sentiment = 'negative'

    def set_country(self, country):
        self.country_prediction = country


    def __str__(self):
        msg = ""
        #msg = "tid: " + self.tid + "\n"
        msg += "content_sentiment: " + self.content_sentiment + "\n"
        msg += "content_geolocation: " + self.content_geolocation + "\n"
        msg += "user_desc: " + self.user_desc + "\n"
        msg += "username: " + self.username + "\n"
        msg += "user_profile_location: " + self.user_profile_location + "\n"
        msg += "tweet_lang: " + self.tweet_lang + "\n"
        msg += "user_lang: " + self.user_lang + "\n"
        msg += "posting_time: " + self.posting_time + "\n"
        if self.time_zone != None:
            msg += "time_zone: " + self.time_zone + "\n"
        else:
            msg += "time_zone: None" + "\n"
        msg += 'geolocation: ' + str(self.country) + "\n"
        msg += 'country_prediction: ' + str(self.country_prediction) + "\n"
        msg += 'predicted sentiment: ' + str(self.sentiment) + "\n"
        msg += 'original sentiment: ' + str(self.original_sentiment) + "\n"
        return msg

    def serialize(self):
        return {
                #'tid' : self.tid,
                #'text' : self.content,
                'sentiment' : self.sentiment,
                'original sentiment' : self.original_sentiment,
                'geolocation' : self.country,
                'country_prediction' : self.country_prediction
                #'created_at' : self.posting_time,
                #'tweet_lang' : self.tweet_lang,
                #'user_lang' : self.user_lang
        }
