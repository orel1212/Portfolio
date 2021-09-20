import json


def parse_tweets(filename):
    dataset = []
    filtered_tweets = 0
    no_lang = 0
    with open(filename) as file:
        for json_tweet in file:
            json_tweet = json.loads(json_tweet)
            print(json_tweet['full_text'])
            sentiment = input('is positive: P/N: ')
            while sentiment != 'P' and sentiment != 'N':
                print("Please input only the correct options")
                print(json_tweet['full_text'])
                sentiment = input('is positive: P/N: ')
            
            if sentiment == 'P':
                json_tweet['sentiment'] = 'positive'
            elif sentiment == 'N':
                json_tweet['sentiment'] = 'negative'
            else: 
                print("ERROR - YOU DIDNT PICK P OR N")
                exit()
            dataset.append(json_tweet)
    return dataset

filename = "../tweets/test"
dataset = parse_tweets(filename+".json")

def save_to_json(data):
    with open(filename+".sentiment.json", 'w') as f:
        for tweet in data:
            #print(json.dumps(tweet))
            f.write(json.dumps(tweet)+'\n')

save_to_json(dataset)