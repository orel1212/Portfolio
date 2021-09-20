#!/bin/bash


deployment_test_data_url="https://www.dropbox.com/sh/sxzyb2inm3uvg5f/AAB_IyYQU3U7sCxP2TN7nNO-a?dl=1"
deployment_metadata_url="https://www.dropbox.com/sh/hn2ue0rx70o8i09/AADdrH0vi2yuG4tCvYgAnB-xa?dl=1"
deployment_serving_url="https://www.dropbox.com/sh/2xcunn2oewppzd8/AADmDKdoCCkTNh0oHJFTngAUa?dl=1"
deployment_demo_url="https://www.dropbox.com/sh/ieg8ik8jpye3ahq/AABP0M1pYWg_eddO6DP5Am4ma?dl=1"

# Downloading the test data
curl $deployment_test_data_url -L -o test_data.zip
mkdir code/api/tests/data
unzip test_data.zip -d code/api/tests/data/ -x /
rm test_data.zip

# Downloading the metadata(country_list, word_to_index_sentiment, etc...)
curl $deployment_metadata_url -L -o metadata.zip
mkdir metadata
unzip metadata.zip -d metadata/ -x /
rm metadata.zip

# Downloading the serving models(Geolocation and Sentiment)
curl $deployment_serving_url -L -o serving.zip
mkdir serving
unzip serving.zip -d serving/ -x /
rm serving.zip

# Downloading the demo tweets
curl $deployment_demo_url -L -o demo.zip
mkdir demo
unzip demo.zip -d demo/ -x /
rm demo.zip




