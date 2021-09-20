# class Config:
#     def __init__(self, host = "localhost", port = "9000"):
#         self.host = host
#         self.port = port
#         self.timesteps = 30
#         self.sequence_content = 30
#         self.sequence_description = 35
#         self.sequence_username = 10
#         self.sequence_profile_location = 8
#         self.num_of_tweet_langs = 34
#         self.num_of_user_langs = 35
#         self.num_of_time_zones = 166
#         self.num_of_posting_timeslots = 144
#         self.num_of_classes = 153
#         self.word_to_index_sentiment_path = "../metadata/word_to_index_sentiment.npy"
#         self.word_to_index_geolocation_path = "../metadata/word_to_index_geolocation.npy"
#         self.country_list_path = "../metadata/country_list"
#         self.time_zone_list_path = "../metadata/time_zone_list"
#         self.tweet_lang_list_path = "../metadata/tweet_lang_list"
#         self.user_lang_list_path = "../metadata/user_lang_list"

# FOR GOOD GEO
class Config:
    def __init__(self, host = "localhost", port = "9000"):
        self.host = host
        self.port = port
        self.timesteps = 30
        self.sequence_content = 30
        self.sequence_description = 35
        self.sequence_username = 10
        self.sequence_profile_location = 8
        self.num_of_tweet_langs = 42
        self.num_of_user_langs = 61
        self.num_of_time_zones = 320
        self.num_of_posting_timeslots = 144
        self.num_of_classes = 190
        self.word_to_index_sentiment_path = "../metadata/word_to_index_sentiment.npy"
        self.word_to_index_geolocation_path = "../metadata/word_to_index_geolocation.npy"
        self.country_list_path = "../metadata/country_list"
        self.time_zone_list_path = "../metadata/time_zone_list"
        self.tweet_lang_list_path = "../metadata/tweet_lang_list"
        self.user_lang_list_path = "../metadata/user_lang_list"
