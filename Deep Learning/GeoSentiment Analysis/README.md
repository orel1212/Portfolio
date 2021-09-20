# Twitter Geosentiment Analysis

A Geosentiment Analysis tool implemented with TensorFlow which uses BiLSTM to train sentiment analysis and CNN to train the geolocation predictions of tweets.

The project includes 4 main components:
- Training a Sentiment Analysis model based on STS(Stanford Twitter Sentiment) corpus <a>http://www.sananalytics.com/lab/twitter-sentiment/</a>
- Training a Geolocation prediction model based on W-NUT (Han et al., 2016) 
- Flask-Python based Restful api that can search for live tweets and analyze them
- An Angular app that uses the Restful api (client) 

## Dependencies
### Sentiment Analysis
* Python 3.5.2
* Tensorflow 1.6.0
    https://www.tensorflow.org/install/
* numpy
    http://www.numpy.org/
* regex
    https://pypi.org/project/regex/
 
 ### Geolocation Prediction
* Python 3.5.2
* Tensorflow 1.6.0
    https://www.tensorflow.org/install/
* numpy
    http://www.numpy.org/
* nltk
    https://www.nltk.org/install.html
* reverse_geocoder
    https://github.com/thampiman/reverse-geocoder
* gensim
    https://radimrehurek.com/gensim/install.html
### Flask Restful api
* Python 3.5.2
* Tensorflow 1.6.0
    https://www.tensorflow.org/install/
* numpy
    http://www.numpy.org/
* Tensorflow Serving
    https://www.tensorflow.org/serving/
* Flask
* Flask-SQLAlchemy
* Flask-JWT-Extended
* gunicorn
* nltk
