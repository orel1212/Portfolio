# Sentiment Analysis
A Sentiment Analysis model that uses BILSTM.

## Dependencies
* Python 3
* Tensorflow
    https://www.tensorflow.org/install/
* numpy
    http://www.numpy.org/
* regex
    https://pypi.org/project/regex/
    
## Installation
1. Install tensorflow https://www.tensorflow.org/install/
2. Install regex `pip3 install regex`
3. run the init_script in order to create the required folders and download the sent 140 datasets and GloVe word embeddigns. `./init_script`

## Usage
### Preprocess
1. run `python3 parse.py` and pick the option that you want to use or you can write your own parser for a different dataset than ours

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
