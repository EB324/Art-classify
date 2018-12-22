import numpy as np
import matplotlib # Solve "RuntimeError" on mac OSX
matplotlib.use('TkAgg') # Solve "RuntimeError" on mac OSX
import matplotlib.pyplot as plt
import os
import cv2
import random
from tqdm import tqdm
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.utils import to_categorical
import pickle
from sklearn.model_selection import train_test_split
import time
from keras.callbacks import TensorBoard

'''Load training data'''

pickle_in = open("trained/X_train.pickle","rb")
X_train = pickle.load(pickle_in)

pickle_in = open("trained/y_train.pickle","rb")
y_train = pickle.load(pickle_in)

pickle_in = open("trained/X_test.pickle","rb")
X_test = pickle.load(pickle_in)

pickle_in = open("trained/y_test.pickle","rb")
y_test = pickle.load(pickle_in)

'''Build CNN model'''
model = Sequential()

input_shape = X_train.shape[1:]
num_classes = 3
dense_layer = 1
layer_size = 256
conv_layer = 2

model.add(Conv2D(layer_size, (3, 3), input_shape=input_shape))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(layer_size, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())

model.add(Dense(layer_size))
model.add(Activation('relu'))

model.add(Dense(num_classes))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        
'''Train model'''

NB_EPOCH = 4
BATCH_SIZE = 32
VERBOSE = 1

model.fit(X_train, y_train, 
          epochs=NB_EPOCH, 
          batch_size=BATCH_SIZE,          
          verbose=VERBOSE)

'''Evaluate model'''

test_loss, test_acc = model.evaluate(X_test, y_test)
print('loss:', test_loss) 
print('accuracy:', test_acc) 

