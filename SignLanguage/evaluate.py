import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
from tensorflow.keras import regularizers
from tensorflow.keras.optimizers import SGD
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tqdm import tqdm
import numpy as np
from PIL import Image
from tensorflow.keras import mixed_precision
import os
import csv
import tensorflow_hub as hub
import dataprocessing

model = tf.keras.models.load_model("MobileNasNet")
label_dict = {}
for i, line in enumerate(open("dataset/wnids.txt", "r")):
    label_dict[line.rstrip("\n")] = int(i)

test_labels = dataprocessing.generate_test_labels("dataset/test/")
test_int = [label_dict[x[0]] for x in test_labels]

batch_size = 100
img_size = 100

pred = dataprocessing.tta_prediction(model,
                                     batch_size,
                                     img_size,
                                     normalize=False,
                                     test_location="dataset/test/")
y_predict_max = np.argmax(pred, axis=1)

print(np.asarray(test_int))
print(y_predict_max)

total = 0.0
correct = 0.0
for x, y in zip(test_int, y_predict_max):
    total += 1
    if x == y:
        correct += 1

accuracy = correct / total

print("Accuracy: " + str(accuracy))