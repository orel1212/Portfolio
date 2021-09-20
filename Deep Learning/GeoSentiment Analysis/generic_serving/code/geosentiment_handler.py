from config import Config
from geolocation_prediction import GeolocationPrediction
from sentiment_analysis import SentimentAnalysis
from tweet import Tweet
import json
from random import shuffle
import argparse
from collections import Counter, OrderedDict
import numpy as np
import matplotlib.pyplot as plt




class GeoSentimentHandler:
    def __init__(self):
        self.config = Config()
        self.sentiment_analysis = SentimentAnalysis(self.config)
        self.geolocation_prediction = GeolocationPrediction(self.config)

    def predict_tweets_sentiment(self,tweets):
        self.sentiment_analysis.predict_tweets_sentiment(tweets, self.config.host, self.config.port)

    def predict_tweets_geolocation(self,tweets):
        self.geolocation_prediction.predict_tweets_geolocation(tweets, self.config.host, self.config.port)



def init_parser():
    epilog_helper_msg="if you decide to supply option, then it supposes that "
    epilog_helper_msg+="you want to create a file with the output of the action of that option."
    epilog_helper_msg+="\nOtherwise it will just load the file.\nFor example:if you specified -p, "
    epilog_helper_msg+="then it will create a scatter plot \nand store the plot in the disk.\nOtherwise, it will won't do it.\n"
    epilog_helper_msg+="Required:\n 1.choose sentiment or geolocation at least to run!\n2.enter file name of the tweets\n3.the file must be in path ../tweets\n4.file type is json\n"
    epilog_helper_msg+="5.scatter plot require geolocation with -g option\n"
    arg_parser = argparse.ArgumentParser(prog='geosentiment_handler.py',
     usage='%(prog)s [options]',
     epilog=epilog_helper_msg,
     formatter_class=argparse.RawTextHelpFormatter)
    arg_parser.add_argument('-f','--file', action='store',help='the name of the file to read the tweets from',dest="file_name")
    arg_parser.add_argument('-s','--sentiment', action='store_true',default=False,help='run the sentiment model on the tweets',dest="sentiment")
    arg_parser.add_argument('-g','--geolocation', action='store_true',default=False,help='run the geolocation model on the tweets',dest="geolocation")
    arg_parser.add_argument('-p','--plot', action='store_true',default=False,help='creating a scatter plot for the geolocation',dest="scatter_plot")
    
    return arg_parser 



def calculate_sentiment_accuracy(tweets):
    correct_pred = 0
    for tweet in tweets:
        if tweet.original_sentiment == tweet.sentiment:
            correct_pred += 1
    acc_sentiment = 100 * correct_pred/len(tweets)
    return acc_sentiment

def calculate_geolocation_accuracy(tweets, neighbours):
    correct_pred = 0
    correct_pred_neighbours = 0

    true_positives = Counter()
    false_negatives = Counter()
    false_positives = Counter()

    for tweet in tweets:
        if tweet.country_prediction in neighbours[tweet.country]:
            correct_pred_neighbours += 1
        elif tweet.country_prediction == tweet.country:
            correct_pred += 1
            true_positives[tweet.country] += 1
        else:
            false_negatives[tweet.country] += 1
            false_positives[tweet.country_prediction] += 1
    acc_geo = 100 * correct_pred/len(tweets)
    acc_geo_n = 100 * (correct_pred+correct_pred_neighbours)/len(tweets)
    return acc_geo, acc_geo_n, true_positives, false_negatives, false_positives

if __name__ == '__main__':
    arg_parser=init_parser()
    flags = vars(arg_parser.parse_args())
    if flags["file_name"] == None:
        print("you must enter file name with -f option! use -h option for help!")
        exit()
    if flags["sentiment"] == False and flags["geolocation"] == False:
        print("you must choose at least sentiment/geolocation with -s/-g (respectively) option! use -h option for help!")
        exit()
    if flags["geolocation"] == False and flags["scatter_plot"] == True:
        print("you must choose geolocation with -g option when using -p! use -h option for help!")
        exit()






    if flags["geolocation"]:
        neighbours = {}
        countries_json = None
        countries_path = "countries.json"
        with open(countries_path) as file:
            countries_json = json.load(file)

        for country in countries_json:
            neighbours[country["alpha_2"]] = country["neighbours"].split(',')
            
    geosentiment_handler = GeoSentimentHandler()
    tweets_sentiment = []
    tweets_geolocation = []
    filename_path= "../tweets/" + flags["file_name"]               
    with open(filename_path) as file:
        for tweet_json in file:
            try:
                tweet_json = json.loads(tweet_json)
                if flags["sentiment"]:
                    tweets_sentiment.append(Tweet(tweet_json))
                if flags["geolocation"] and tweet_json["place"] is not None and tweet_json["place"]["country_code"] != "": 
                    tweets_geolocation.append(Tweet(tweet_json))
            except ValueError as e:
                print('invalid file type: json Required')
                exit()

    if flags["sentiment"]:
        geosentiment_handler.predict_tweets_sentiment(tweets_sentiment)
        acc_sentiment = calculate_sentiment_accuracy(tweets_sentiment)
        print("number of tweets for sentiment: ", len(tweets_sentiment))
        print("sentiment acc: ", acc_sentiment)

    if flags["geolocation"]:

        tweets_geolocation = tweets_geolocation[:30000]
        geosentiment_handler.predict_tweets_geolocation(tweets_geolocation)
        acc_geo, acc_geo_n, true_positives, false_negatives, false_positives = calculate_geolocation_accuracy(tweets_geolocation, neighbours)


        temp = {}
        for x,y in true_positives.items():
            if y > 5:
                temp[x] = y

        true_positives = temp

        
        print("number of tweets for geolocation: ", len(tweets_geolocation))
        print("geolocation acc: ", acc_geo)
        print("geolocation acc with neighbours: ", acc_geo_n)

        if flags["scatter_plot"]:
            precision = {}
            recall = {}
            area = {}
            for key, tp in true_positives.items():
                precision[key] = round(tp / (tp + false_positives[key]), 2)
                area[key] = tp + false_positives[key]
                recall[key] = round(tp / (tp + false_negatives[key]), 2)


        

            ordered_cc = list(OrderedDict(sorted(precision.items())))
            ordered_precision = list(OrderedDict(sorted(precision.items())).values())
            ordered_recall = list(OrderedDict(sorted(recall.items())).values())
            ordered_area = np.array(list(OrderedDict(sorted(area.items())).values()))
            s = [10*2**n for n in np.log(ordered_area)]

            plt.scatter(ordered_precision, ordered_recall, s = s, alpha = 0.5)
            plt.title(flags["file_name"] + "precision/recall")
            plt.xlabel("precision")
            plt.ylabel("recall")
            for i, cc in enumerate(ordered_cc):
                plt.annotate(cc, (ordered_precision[i],ordered_recall[i]), ha='center')
            plt.savefig(flags["file_name"]+'.png')
            #plt.show()
