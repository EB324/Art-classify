import numpy as np
# import matplotlib # Solve "RuntimeError" on mac OSX
# matplotlib.use('TkAgg') # Solve "RuntimeError" on mac OSX
import matplotlib.pyplot as plt
import os
import cv2
import random
import pickle
from tqdm import tqdm
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split

'''Load Dataset'''

DATADIR = 'dataset/'
CATEGORIES = ['iconography', 'painting', 'sculpture']
IMG_SIZE = 50

dataset = []

def create_dataset():
    for category in CATEGORIES:  
        
        path = os.path.join(DATADIR,category)  # create path 
        class_num = CATEGORIES.index(category)  # get the classification 

        for img in tqdm(os.listdir(path)):  # iterate over each image 
            try:
                img_array = cv2.imread(os.path.join(path,img),
                                       cv2.IMREAD_GRAYSCALE)  # convert to array
                new_array = cv2.resize(img_array, 
                                        (IMG_SIZE, IMG_SIZE))  # resize to normalize data size
                dataset.append([new_array, 
                                class_num])  # add this to our training_data
            except Exception as e:  
                pass

create_dataset()        

random.shuffle(dataset) # shuffle data

'''Create Variables'''

X = []
y = []

for features,label in dataset:
    X.append(features)
    y.append(label)

X = np.array(X).reshape(-1, 
                        IMG_SIZE, IMG_SIZE, 
                        1)

# Split into train and test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2) 

# Normalization
X_train = X_train/255.0 
X_test = X_test/255.0

# Convert class vector to binary class matrix
y_train = to_categorical(y_train, num_classes=3)
y_test = to_categorical(y_test, num_classes=3)

'''Build CNN Model'''

input_shape = X_train.shape[1:]
num_classes = 3
dense_layer = 1
layer_size = 256
conv_layer = 2

model = Sequential()

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

'''Compile the Model'''

model.compile(loss='categorical_crossentropy', 
              optimizer='adam', 
              metrics=['accuracy'])

'''Train the Model'''

NB_EPOCH = 4
BATCH_SIZE = 32
VERBOSE = 1

model.fit(X_train, y_train, 
          epochs=NB_EPOCH, 
          batch_size=BATCH_SIZE,          
          verbose=VERBOSE)

'''Evaluate the Model'''

test_loss, test_acc = model.evaluate(X_test, y_test)
print('loss:', test_loss) 
print('accuracy:', test_acc)

