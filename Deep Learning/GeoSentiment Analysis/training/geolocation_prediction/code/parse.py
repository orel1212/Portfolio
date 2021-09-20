import json
import numpy as np
import reverse_geocoder as rg


def parse_tweets(filename):
    dataset = []
    filtered_tweets = 0
    coordinates = []
    with open(filename) as file:
        for json_tweet in file:
            tweet = []
            json_tweet = json.loads(json_tweet)
            if json_tweet["coordinates"] is not None:
                tweet.append(json_tweet["text"])
                tweet.append(json_tweet["user"]["description"])
                tweet.append(json_tweet["user"]["name"])#used to be screen name         
                tweet.append(json_tweet["user"]["location"])
                tweet.append(json_tweet["lang"])
                tweet.append(json_tweet["user"]["lang"])

                if json_tweet["user"]["time_zone"] is not None:
                    tweet.append(json_tweet["user"]["time_zone"])
                else:
                    tweet.append("und")
                tweet.append(json_tweet["created_at"])
                # [longitude, latitude]
                lon = json_tweet["coordinates"]["coordinates"][0] 
                lat = json_tweet["coordinates"]["coordinates"][1]
                coordinates.append((lat,lon))
                dataset.append(tweet)
            else:
                filtered_tweets += 1
        labels = rg.search(coordinates)

        for i, label in enumerate(labels):
            dataset[i].insert(0,label["cc"])

    print("filtered_tweets: ", filtered_tweets)
    print("added_tweets: ", len(dataset))
    return dataset


def get_categories_labels(filename, categories_list):
    filtered_tweets = 0
    coordinates = []
    with open(filename) as file:
        for json_tweet in file:
            tweet = []
            json_tweet = json.loads(json_tweet)
            if json_tweet["coordinates"] is not None: 
                categories_list[0].add(json_tweet["lang"])
                categories_list[1].add(json_tweet["user"]["lang"])
                if json_tweet["user"]["time_zone"] is not None:
                    categories_list[2].add(json_tweet["user"]["time_zone"])   
                # [longitude, latitude]
                lon = json_tweet["coordinates"]["coordinates"][0] 
                lat = json_tweet["coordinates"]["coordinates"][1]
                coordinates.append((lat,lon))
            else:
                filtered_tweets += 1
        labels = rg.search(coordinates)
        
        for label in labels:
            categories_list[3].add(label["cc"])

def save_to_json(save_path, data):
    with open(save_path, 'w') as outfile:
        json.dump(data, outfile)







print("Press 1 to create the categories list\nPress 2 to parse the datasets")
option = input("pick your option: ")
try:
    option = int(option)
except:
    print("Please enter a number")
    exit()
if option == 1:
    get_categories_flag = True
elif option == 2:
    get_categories_flag = False
else:
    print("Please pick one of the options above")
    exit()

path = "../datasets/wnut/"

if get_categories_flag:
    country_set = set()
    user_lang_set = set()
    user_lang_set.add("und")
    tweet_lang_set = set()
    tweet_lang_set.add("und")
    time_zone_set = set()
    time_zone_set.add("und")
    categories_list = [tweet_lang_set, user_lang_set, time_zone_set, country_set]

    test_file = path + "test.tweet.output.json"
    get_categories_labels(test_file, categories_list)
    training_file = path + "training.output.json"
    get_categories_labels(training_file, categories_list)
    valid_file = path + "validation.tweet.output.json"
    get_categories_labels(valid_file, categories_list)

    country_list = list(country_set)
    country_list.sort()
    user_lang_list = list(user_lang_set)
    user_lang_list.sort()
    tweet_lang_list = list(tweet_lang_set)
    tweet_lang_list.sort()
    time_zone_list = list(time_zone_set)
    time_zone_list.sort()
    
    save_to_json("../datasets/wnut/country_list",country_list)
    save_to_json("../datasets/wnut/user_lang_list",user_lang_list)
    save_to_json("../datasets/wnut/tweet_lang_list",tweet_lang_list)
    save_to_json("../datasets/wnut/time_zone_list",time_zone_list)
    
    print("done saving lists")

else:
    training_file = path + "training.output.json"
    save_train_path = "../datasets/wnut/train_set"

    valid_file = path + "validation.tweet.output.json"
    save_valid_path = "../datasets/wnut/valid_set"

    test_file = path + "test.tweet.output.json"
    save_test_path = "../datasets/wnut/test_set"
    
    save_to_json(save_train_path,parse_tweets(training_file))
    save_to_json(save_valid_path,parse_tweets(valid_file))
    save_to_json(save_test_path,parse_tweets(test_file))


