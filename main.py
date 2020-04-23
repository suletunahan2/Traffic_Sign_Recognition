# -*- coding: utf-8 -*-
"""
Created on Fri Apr  3 21:11:35 2020

@author: suletunahan2
"""

import pickle
import numpy as np

# Load pickled data
import pickle
import numpy as np

training_file = '/content/drive/My Drive/Colab Notebooks/CNN/traffic-signs-data2.zip (Unzipped Files)/train.p'
testing_file = '/content/drive/My Drive/Colab Notebooks/CNN/traffic-signs-data2.zip (Unzipped Files)/test.p'

with open(training_file,
          mode='rb') as f:  # rb:Opens a file for reading only in binary format. The file pointer is placed at the beginning of the file. This is the default mode.
    train = pickle.load(f)
with open(testing_file, mode='rb') as f:
    test = pickle.load(f)

x_train, y_train = train['features'], train['labels']
x_test, y_test = test['features'], test['labels']

print('Reading data done!')