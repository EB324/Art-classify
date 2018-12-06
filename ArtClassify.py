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


DATADIR = 'dataset/training_set/'
CATEGORIES = ['iconography', 'painting', 'sculpture']
IMG_SIZE = 50

'''Build training data'''

training_data = []

def create_training_data():
    for category in CATEGORIES:  
        
        path = os.path.join(DATADIR,category)  # create path 
        class_num = CATEGORIES.index(category)  # get the classification 

        for img in tqdm(os.listdir(path)):  # iterate over each image 
            try:
                img_array = cv2.imread(os.path.join(path,img),
                                       cv2.IMREAD_GRAYSCALE)  # convert to array
                new_array = cv2.resize(img_array, 
                                        (IMG_SIZE, IMG_SIZE))  # resize to normalize data size
                training_data.append([new_array, 
                                      class_num])  # add this to our training_data
            except Exception as e:  
                pass

create_training_data()        

'''Shuffle data'''

random.shuffle(training_data)

'''Generate training data'''

X = []
y = []

for features,label in training_data:
    X.append(features)
    y.append(label)

X = np.array(X).reshape(-1, 
                        IMG_SIZE, IMG_SIZE, 
                        1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.8)

'''Build CNN model'''

# Normalization
X_train = X_train/255.0
X_test = X_test/255.0

# Variables
num_classes = 3
input_shape = X_train.shape[1:]

# Build model
model = Sequential()

# Layer 1
model.add(Conv2D(32, kernel_size=(5, 5), strides=(1, 1),
                 activation='relu',
                 input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

# Layer 2
model.add(Conv2D(64, (5, 5), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(1000, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))

# Compile model
model.compile(loss='categorical_crossentropy',
              optimizer='SGD',
              metrics=['accuracy'])

'''Train model'''

NB_EPOCH = 100
BATCH_SIZE = 20
VERBOSE = 1

y_train = to_categorical(y_train, num_classes=3) #Converts class vector to binary class matrix.
y_test = to_categorical(y_test, num_classes=3)

model.fit(X_train, y_train, 
          epochs=NB_EPOCH, 
          batch_size=BATCH_SIZE,          
          verbose=VERBOSE)

pickle_out = open('artClassify.pickle','wb')
pickle.dump(model,pickle_out)
pickle_out.close()

val_loss, val_acc = model.evaluate(X_test, y_test, batch_size=20)
print('loss:', val_loss) 
print('accuracy:', val_acc) 