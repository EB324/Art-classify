import numpy as np
#import matplotlib # Solve "RuntimeError" on mac OSX
#matplotlib.use('TkAgg') # Solve "RuntimeError" on mac OSX
import matplotlib.pyplot as plt
import os
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.utils import to_categorical
import pickle
import time
from keras.callbacks import TensorBoard

'''Load variables'''

pickle_in = open("trained/X_train.pickle","rb")
X_train = pickle.load(pickle_in)

pickle_in = open("trained/y_train.pickle","rb")
y_train = pickle.load(pickle_in)

pickle_in = open("trained/X_test.pickle","rb")
X_test = pickle.load(pickle_in)

pickle_in = open("trained/y_test.pickle","rb")
y_test = pickle.load(pickle_in)

'''Build and train CNN model'''

num_classes = 3
input_shape = X_train.shape[1:]

dense_layers = [0, 1, 2]
layer_sizes = [32, 64, 128, 256]
conv_layers = [1, 2, 3]

for dense_layer in dense_layers:
    for layer_size in layer_sizes:
        for conv_layer in conv_layers:
            NAME = "{}-conv-{}-nodes-{}-dense-{}".format(conv_layer, layer_size, dense_layer, int(time.time()))
            print(NAME)

            model = Sequential()

            model.add(Conv2D(layer_size, (3, 3), input_shape=input_shape))
            model.add(Activation('relu'))
            model.add(MaxPooling2D(pool_size=(2, 2)))

            for l in range(conv_layer-1):
                model.add(Conv2D(layer_size, (3, 3)))
                model.add(Activation('relu'))
                model.add(MaxPooling2D(pool_size=(2, 2)))

            model.add(Flatten())

            for _ in range(dense_layer):
                model.add(Dense(layer_size))
                model.add(Activation('relu'))

            model.add(Dense(num_classes))
            model.add(Activation('softmax'))

            tensorboard = TensorBoard(log_dir="logs/{}".format(NAME))

            model.compile(loss='categorical_crossentropy',
                          optimizer='adam',
                          metrics=['accuracy'])

            model.fit(X_train, y_train,
                      batch_size=32,
                      epochs=20,
                      validation_split=0.3,
                      callbacks=[tensorboard])
