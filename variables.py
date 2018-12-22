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

'''Generate and normalize variables'''

X = []
y = []

for features,label in training_data:
    X.append(features)
    y.append(label)

X = np.array(X).reshape(-1, 
                        IMG_SIZE, IMG_SIZE, 
                        1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

X_train = X_train/255.0 # Normalization
X_test = X_test/255.0

y_train = to_categorical(y_train, num_classes=3) # Converts class vector to binary class matrix.
y_test = to_categorical(y_test, num_classes=3)

pickle_out = open("X_train.pickle","wb") # Save X_train into pickle files
pickle.dump(X_train, pickle_out)
pickle_out.close()

pickle_out = open("y_train.pickle","wb") # Save y_train into pickle files
pickle.dump(y_train, pickle_out)
pickle_out.close()

pickle_out = open("X_test.pickle","wb") # Save X_test into pickle files
pickle.dump(X_test, pickle_out)
pickle_out.close()

pickle_out = open("y_test.pickle","wb") # Save y_test into pickle files
pickle.dump(y_test, pickle_out)
pickle_out.close()
