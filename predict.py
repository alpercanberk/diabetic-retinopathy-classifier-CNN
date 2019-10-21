
import os
from keras.models import Sequential, model_from_json
from keras.optimizers import SGD, RMSprop, Adam
from keras.layers import Dense, Activation, Dropout, BatchNormalization, Conv2D, Flatten, MaxPooling2D
import numpy as np

json_file = open('../model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)

# load weights into new model
loaded_model.load_weights("../model.h5")
print("Loaded model from disk")

#let the predictions begin
loaded_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

X_test  = np.load('../processed_X_test.npy')

predictions = loaded_model.predict(X_test, verbose=1)

#save the results in a file
np.save("../predictions.npy", predictions)
