import matplotlib.image as mpimg
import matplotlib.pyplot as plt

import numpy as np

from tempfile import TemporaryFile

import os
from os import listdir
from os.path import isfile, join

from PIL import Image, ImageEnhance

import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder


TRAIN_FOLDER_PATH = "../train_images/"

training_examples = os.listdir('../train_images')

MAX_TRAINING_EXAMPLES = len(training_examples)

print("Number of training examples:", len(training_examples))

y_df= pd.read_csv('../train.csv')

X_train = np.zeros((MAX_TRAINING_EXAMPLES, 500, 650, 3))
y_train = np.zeros((MAX_TRAINING_EXAMPLES, 5))

for i in range(0, MAX_TRAINING_EXAMPLES):

	file = TRAIN_FOLDER_PATH + training_examples[i]
	print("Opening file:", file)
	img = Image.open(file)

	resized_img = img.resize((750, 500))
	cropped_img = np.asarray(resized_img)[:,50:-50,:]
	X_train[i, :, :, :] = cropped_img

	# plt.imshow(cropped_img)
	# plt.show()

	diagnosis = int(y_df[y_df['id_code'] == training_examples[i][:-4]]['diagnosis'].iat[0])
	print(str((i/MAX_TRAINING_EXAMPLES)*100) + "%")
	y_train[i, diagnosis] = 1

np.save("../processed_X_train.npy", X_train)
np.save("../processed_y_train.npy", y_train)

print(X_train.shape)
print(y_train.shape)


# i = int(y_train[y_train['id_code'] == '001639a390f0']['diagnosis'].iat[0])
# print(type(int(i)))

# ['diagnosis'].iat[0]
