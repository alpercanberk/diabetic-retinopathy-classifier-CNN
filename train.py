import matplotlib.image as mpimg
import matplotlib.pyplot as plt

import numpy as np

from tempfile import TemporaryFile

import os
from os import listdir
from os.path import isfile, join

from PIL import Image

from keras.models import Sequential, model_from_json
from keras.optimizers import SGD, RMSprop, Adam
from keras.layers import Dense, Activation, Dropout, BatchNormalization, Conv2D, Flatten, MaxPooling2D



X_data = np.load('../processed_X_train.npy')
y_data = np.load('../processed_y_train.npy')

X_train = X_data[:int(len(X_data)*0.9),:,:,:]
y_train = y_data[:int(len(X_data)*0.9),:]

X_validation = X_data[int(len(X_data)*0.9):,:,:,:]
y_validation = y_data[int(len(X_data)*0.9):,:]

def create_model():
    cnn = Sequential()

    cnn.add(Conv2D(32, (3,3), strides=(1, 1),
                     padding='valid',
                     use_bias=True,
                     kernel_initializer='glorot_uniform',
                     bias_initializer='zeros'))
    cnn.add(Activation('relu'))
    cnn.add(MaxPooling2D(pool_size = (2, 2), strides=2))

    cnn.add(Conv2D(64, (3,3), strides=(1, 1),
                     padding='valid',
                     use_bias=True,
                     kernel_initializer='glorot_uniform',
                     bias_initializer='zeros'))
    cnn.add(Activation('relu'))
    cnn.add(MaxPooling2D(pool_size = (2, 2), strides=2))

    cnn.add(Conv2D(128, (3,3), strides=(1, 1),
                     padding='valid',
                     use_bias=True,
                     kernel_initializer='glorot_uniform',
                     bias_initializer='zeros'))
    cnn.add(Activation('relu'))
    cnn.add(MaxPooling2D(pool_size = (2, 2), strides=2))

    cnn.add(Flatten())
    cnn.add(Dense(256))
    cnn.add(Activation('relu'))
    cnn.add(Dropout(0.25))
    cnn.add(BatchNormalization())

    cnn.add(Dense(64))
    cnn.add(Activation('relu'))
    cnn.add(Dropout(0.25))

    cnn.add(Dense(5))
    cnn.add(Activation('softmax'))
    return cnn

model = create_model()
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(X_train, y_train, epochs=1, verbose=1)

score, acc = model.evaluate(X_train, y_train)
score, acc = model.evaluate(X_validation, y_validation)

print('Train accuracy:', acc)
print('Validation accuracy:', acc)

model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("../model.h5")
print("Saved model to disk")

# score, acc = model.evaluate(X_train.values, y_train.values)
# score, acc = model.evaluate(X_test.values, y_test.values)
#
# print('Train accuracy:', acc)
# print('Test accuracy:', acc)
#
# return model
