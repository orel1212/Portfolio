# Twitter Geosentiment RESTful API
A RESTful API server which handles different requests and interacts with our Geolocation prediction trained models in order to predict the sentiment and location of tweets.

## Dependencies
* Python 3.6
* Tensorflow
* Numpy
* Flask
* Flask-SQLAlchemy
* Flask-JWT-Extended
* regex `pip3 install regex`
* nltk `pip3 install nltk`
* tweepy `pip3 install tweepy`
* nose2 ( for unit tests) `pip3 install nose2`

## USAGE
Each endpoints returns a result in JSON
The endpoints are documented here: http://geosentiment-analysis.s3-website-us-east-1.amazonaws.com/api
