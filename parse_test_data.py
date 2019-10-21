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

TEST_FOLDER_PATH = "../test_images/"

test_files = os.listdir('../test_images')

print("Number of test files:", len(test_files))

y_df = pd.read_csv('../train.csv')

X_test = np.zeros((len(test_files), 500, 650, 3))

for i in range(0, len(test_files)):

    file = TEST_FOLDER_PATH + test_files[i]
    print("Opening file:", file)
    img = Image.open(file)
    print(str(i), " out of ",str(len(test_files)))


    resized_img = img.resize((750, 500))
    cropped_img = np.asarray(resized_img)[:,50:-50,:]
    X_test[i, :, :, :] = cropped_img

np.save("../processed_X_test.npy", X_test)
