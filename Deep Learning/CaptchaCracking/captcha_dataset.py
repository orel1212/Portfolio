import torch
from torch.utils.data import Dataset
from PIL import Image

import glob
import os

import numpy as np

MIN_LEN = 4
MAX_LEN = 6

class CaptchaDataSet(Dataset):
  def __init__(self, path, transform, image_extension = 'png'):
    self.transform = transform
    self.img_dir_path = path
    print("I am nig")
    self.all_imgs_path = glob.glob(os.path.join(self.img_dir_path, '*'+image_extension))
    print(" I am nigga")
    self.blank_char = '_'
    self.end_sentence = '>'
    self.classes = self.blank_char + self.end_sentence + '0123456789abcdefghijklmnopqrstuvwxyz'
    self.real_classes = self.blank_char + self.end_sentence + '345689abcdehjkmnprstuvwxy'
    self.max_len = MAX_LEN
    self.real_flag = True

  def __len__(self):
    return len(self.all_imgs_path)

  def __getitem__(self, index):

    img_path = self.all_imgs_path[index]

    image = Image.open(img_path)
    image = image.convert('RGB')

    t_image = self.transform(image)


    label = ''.join(img_path.split('/')[-1].split('.')[0]).lower()
    idxes = []
    if self.real_flag == True:
      try:
        idxes = [self.real_classes.index(letter) for letter in label] + (self.max_len - len(label)) * [
          self.real_classes.index(self.end_sentence)]
      except:
        print("label: " + label)
        print("the label has letters not in real_classes!")
    else:
      idxes = [self.classes.index(letter) for letter in label] + (self.max_len - len(label)) * [
        self.classes.index(self.end_sentence)]

    tensor_label = torch.LongTensor(idxes)
    return t_image, tensor_label, torch.LongTensor([len(label)]), np.array(image)

  def get_num_classes(self):
    if self.real_flag == True:
      return len(self.real_classes)
    else:
      return len(self.classes)

  def get_end_sentence_idx(self):
    if self.real_flag == True:
      return self.real_classes.index(self.end_sentence)
    else:
      return self.classes.index(self.end_sentence)

  def get_blank_idx(self):
    if self.real_flag == True:
      return self.real_classes.index(self.blank_char)
    else:
      return self.classes.index(self.blank_char)

  def get_classes(self):
    if self.real_flag == True:
      return self.real_classes
    else:
      return self.classes