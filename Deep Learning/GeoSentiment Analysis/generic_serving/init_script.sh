#!/bin/bash


generic_serving_metadata_url="https://www.dropbox.com/sh/hn2ue0rx70o8i09/AADdrH0vi2yuG4tCvYgAnB-xa?dl=1"
generic_serving_serving_url="https://www.dropbox.com/sh/2xcunn2oewppzd8/AADmDKdoCCkTNh0oHJFTngAUa?dl=1"
generic_serving_tweets_url="https://www.dropbox.com/sh/ieg8ik8jpye3ahq/AABP0M1pYWg_eddO6DP5Am4ma?dl=1"

# Downloading the metadata(country_list, word_to_index_sentiment, etc...)
curl $generic_serving_metadata_url -L -o metadata.zip
mkdir metadata
unzip metadata.zip -d metadata/ -x /
rm metadata.zip

# Downloading the serving models(Geolocation and Sentiment)
curl $generic_serving_serving_url -L -o serving.zip
mkdir serving
unzip serving.zip -d serving/ -x /
rm serving.zip

# Downloading the demo tweets
curl $generic_serving_tweets_url -L -o tweets.zip
mkdir tweets
unzip tweets.zip -d tweets/ -x /
rm tweets.zip




