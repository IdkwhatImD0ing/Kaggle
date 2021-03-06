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
import pandas as pd

import dataprocessing

keras.backend.set_image_data_format('channels_first')


def run():
    label_dict = {}
    for i, line in enumerate(open("Dataset/wnids.txt", "r")):
        label_dict[line.rstrip("\n")] = int(i)

    batch_size = 64
    img_size = 100
    num_classes = 26
    ### PARSING TRAIN/VALDIATION FILES
    train_data, val_data, test_data = dataprocessing.get_data(
        batch_size, img_size)

    ### Optimized Neural Network
    model = keras.models.Sequential()

    # Model Layers
    model.add(keras.layers.Input(shape=(3, img_size, img_size)))
    model.add(keras.layers.Rescaling(1. / 255))
    model.add(
        keras.layers.Conv2D(filters=32,
                            kernel_size=(3, 3),
                            strides=(1, 1),
                            padding="same",
                            activation='relu'))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.MaxPooling2D((2, 2)))
    model.add(
        keras.layers.Conv2D(filters=64,
                            kernel_size=(3, 3),
                            strides=(1, 1),
                            padding="same",
                            activation='relu'))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.MaxPooling2D((2, 2)))

    model.add(
        keras.layers.Conv2D(filters=128,
                            kernel_size=(3, 3),
                            padding="same",
                            activation='relu'))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.MaxPooling2D((2, 2)))
    model.add(
        keras.layers.Conv2D(filters=128,
                            kernel_size=(3, 3),
                            padding="same",
                            activation='relu'))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.MaxPooling2D((2, 2)))

    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(512, activation='swish'))
    model.add(keras.layers.Dropout(0.2))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Dense(256, activation='swish'))
    model.add(keras.layers.Dropout(0.2))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Dense(num_classes, activation='softmax'))

    model.build(input_shape=(None, 3, img_size, img_size))
    model.summary()
    model.compile(optimizer=tf.keras.optimizers.Adam(),
                  loss=tf.keras.losses.CategoricalCrossentropy(),
                  metrics=['accuracy'])

    num_epochs = 100
    lr_reduction = keras.callbacks.ReduceLROnPlateau(monitor='val_loss',
                                                     patience=4,
                                                     verbose=1,
                                                     factor=0.4,
                                                     min_lr=0.0001)

    early_stop = keras.callbacks.EarlyStopping(monitor='val_loss',
                                               min_delta=0.00001,
                                               patience=8,
                                               mode='auto',
                                               restore_best_weights=True)

    history = model.fit(train_data,
                        workers=16,
                        epochs=num_epochs,
                        validation_data=val_data,
                        batch_size=batch_size,
                        verbose=1,
                        callbacks=[early_stop, lr_reduction],
                        max_queue_size=30)

    model.evaluate(test_data)
    model.save("ASLMOdel")

    plt.plot(history.history['accuracy'], label='accuracy')
    plt.plot(history.history['val_accuracy'], label='val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.xlim([1, 100])
    plt.ylim([0.1, 1.0])
    plt.legend(loc='lower right')
    plt.show()


run()