# Twitter Geosentiment Analysis

A Geosentiment Analysis tool implemented with TensorFlow which uses BiLSTM to train sentiment analysis and CNN to train the geolocation predictions of tweets.

The project includes 4 main components:
- Training a Sentiment Analysis model based on STS(Stanford Twitter Sentiment) corpus
- Training a Geolocation prediction model based on W-NUT 2016 (Han et al., 2016) 
- Flask-Python based Restful api that can search for live tweets and analyze them
- An Angular app that uses the Restful api (client) 

## Deep learning
### Sentiment Analysis
#### Reference
Hochreiter, Sepp, and Jürgen Schmidhuber. "Long short-term memory." Neural computation 9.8 (1997): 1735-1780.
#### Architecture
![alt tag](https://github.com/orel1212/MyWorks/blob/main/Angular%20%2B%20Flask%20%2B%20Deep%20Learning/GeoSentiment%20Analysis/bilstm.png)
#### Dataset
 STS(Stanford Twitter Sentiment) corpus <a>http://www.sananalytics.com/lab/twitter-sentiment/</a><br>
 1.6M tweets -> 800k positive and 800k negative
#### Word Embedding
GloVe - trained on 2B tweets from twitter (1.2m is the size of the vocab)
#### Training
we used l2 norm and gradient clipping for mitigations.
#### Test
84.95% on real tweets from twitter (259 live tweets)

### Geolocation Prediction
#### Reference
Huang, Binxuan, and Kathleen M. Carley. "On Predicting Geolocation of Tweets using Convolutional Neural Networks." arXiv preprint arXiv:1704.05146 (2017).
#### Architecture
![alt tag](https://github.com/orel1212/MyWorks/blob/main/Angular%20%2B%20Flask%20%2B%20Deep%20Learning/GeoSentiment%20Analysis/cnn.png)
#### Dataset
W-NUT 2016 (Han et al., 2016) <br>
12.8M tweets -> only 8.1M tweets were available
#### Word Embedding
Word2Vec - trained on vector of size 300 features
#### Training
we used dropout regulator and gradient clipping for mitigations.
#### Test
91.47% on real tweets from twitter (3130 live tweets)

## Angular for client
![alt tag](https://github.com/orel1212/MyWorks/blob/main/Angular%20%2B%20Flask%20%2B%20Deep%20Learning/GeoSentiment%20Analysis/main.png)
![alt tag](https://github.com/orel1212/MyWorks/blob/main/Angular%20%2B%20Flask%20%2B%20Deep%20Learning/GeoSentiment%20Analysis/login.png)
![alt tag](https://github.com/orel1212/MyWorks/blob/main/Angular%20%2B%20Flask%20%2B%20Deep%20Learning/GeoSentiment%20Analysis/online_search.png)
![alt tag](https://github.com/orel1212/MyWorks/blob/main/Angular%20%2B%20Flask%20%2B%20Deep%20Learning/GeoSentiment%20Analysis/demo_search.png)
![alt tag](https://github.com/orel1212/MyWorks/blob/main/Angular%20%2B%20Flask%20%2B%20Deep%20Learning/GeoSentiment%20Analysis/hashtag_cloud.png)

### Colored Map and table for geosentiment statistics
![alt tag](https://github.com/orel1212/MyWorks/blob/main/Angular%20%2B%20Flask%20%2B%20Deep%20Learning/GeoSentiment%20Analysis/map_statistics.png)
![alt tag](https://github.com/orel1212/MyWorks/blob/main/Angular%20%2B%20Flask%20%2B%20Deep%20Learning/GeoSentiment%20Analysis/table_statistics.png)

## Folder Tree
deployment -> Flask + Gunicorn + Deployed models <br>
frontend -> Angular <br>
training -> geolocation + sentiment analysis tensorflow based training code <br>


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

## Made by Orel Lavie and Timor Satarov
