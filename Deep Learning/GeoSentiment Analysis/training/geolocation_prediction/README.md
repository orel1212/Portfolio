# Geolocation Prediction
A Geolocation Prediction model based on the following paper: https://arxiv.org/abs/1704.05146
## Dependencies
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

## Installation
1. Install tensorflow https://www.tensorflow.org/install/
2. Install nltk `pip3 install nltk`
3. install reverse_geocoder https://github.com/thampiman/reverse-geocoder `pip3 install reverse_geocoder`
4. install gensim: `pip3 install gensim`
5. run the init_script in order to create the required folders and download the Word2Vec. `./init_script`

## Dataset Information
Each Dataset is in the following jsonlines format
```
{
    "user": 
        {"name": "X", 
        "time_zone": "und", 
        "location": "location", 
        "lang": "en", 
        "description": "user_desc"
    }, 
    "created_at": "Thu Feb 06 03:39:20 +0000 2014", 
    "text": "user_text", 
    "lang": "en", 
    "coordinates": {"coordinates": [-23.90914142, 32.98013331]}
}
```

## Usage
### Preprocess
1. First, replace the default files path in the file paree.py to your train, validation and test datasets.
2. run `python3 parse.py` and pick the first option in order to parse your dataset
3. run `python3 parse.py` and pick the second option in order to create the metadata for the datasets(a list of countries, timezones and more...)

### Training
```
usage: main.py [options]

optional arguments:
  -h, --help            show this help message and exit
  -p, --preprocess      preprocessing the data set
  -we, --word_embeds    creating the embeddings of the set
  -im, --ids_matrix     creating the ids matrix of the set
  -lt, --load_train     loading latest saved train model to train
  -lv, --load_valid     loading latest saved valid model to train
  -t, --test_model      test loaded model
  -d DIR_NAME, --directory DIR_NAME
                        directory name of the logs and the model
  -s SERVING_VERSION, --serving SERVING_VERSION
                        version of the model as integer exported by tf_serving

if you decide to supply option, then it supposes that you want to create a file with the output of the action of that option.
Otherwise it will just load the file.
For example:if you specified -p, then it will parse the dataset 
and store the file in the disk.
Otherwise, it will just load the file.
```

1. First you have to change the default file paths in the config.py file.
2. run `python3 main.py -d dir_name -p` in order to preprocess the datasets and start the training process
After training the model, you can load it again and train it more, test it or export it for serving

