# Captcha Cracking

A Captcha cracking tool based on CNN and LSTM


## Dataset
109K captcha images with labels (names) <br>
<a> https://drive.google.com/file/d/1nalIGeKAJk9OaFrmLALEJC56lAxyE7K6/view </a>

## Model
CNN + LSTM - > CRNN text recognition <br>
<a> https://dl.acm.org/doi/abs/10.1145/3297067.3297073 </a>

## Train + Test
0.8/0.2 split <br>

99.33% Accuracy on 21838 samples (20% test)

## Error predictions samples
16 samples chosen as an example to show how the model misclassifies <br>
![alt tag](https://github.com/orel1212/MyWorks/blob/main/Deep%20Learning/captcha_errors.JPG)

## Dependencies
Python 3.8
Pytorch
Numpy

## Made by Orel Lavie and Tamir Gabay