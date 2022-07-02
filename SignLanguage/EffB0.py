import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
from tensorflow.keras import regularizers
from tensorflow.keras.optimizers import SGD
from sklearn.model_selection import train_test_split
from tensorflow import keras
from keras.layers.advanced_activations import PReLU
from tqdm import tqdm
import numpy as np
from PIL import Image
from tensorflow.keras import mixed_precision
import os
import csv
import tensorflow_hub as hub

import dataprocessing

policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)


def run():
    label_dict = {}
    for i, line in enumerate(open("dataset/wnids.txt", "r")):
        label_dict[line.rstrip("\n")] = int(i)

    batch_size = 64
    img_size = 200

    num_classes = 27
    ### PARSING TRAIN/VALDIATION FILES
    train_data, val_data = dataprocessing.generate_augmented_images(
        batch_size, img_size, normalize=False)

    ### PARSING TEST IMAGES
    test_labels = dataprocessing.generate_test_labels()
    test_int = [label_dict[x.replace(".jpg", "")] for x in test_labels]

    effModel = keras.applications.EfficientNetB0(weights='imagenet',
                                                 pooling='avg',
                                                 include_top=False,
                                                 input_shape=(img_size,
                                                              img_size, 3))

    ### Optimized Neural Network
    model = keras.models.Sequential()

    # Model Layers
    model.add(effModel)
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(1024, activation='swish'))
    model.add(keras.layers.Dense(256, activation='swish'))
    model.add(keras.layers.Dense(num_classes, activation='softmax'))

    model.build(input_shape=(None, img_size, img_size, 3))
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

    model.evaluate(val_data)

    ##Matching Predictions with Correct Image ID
    pred = dataprocessing.tta_prediction(model,
                                         batch_size,
                                         img_size,
                                         normalize=False)
    y_predict_max = np.argmax(pred, axis=1)

    total = 0.0
    correct = 0.0
    for x, y in zip(test_int, y_predict_max):
        total += 1
        if x == y:
            correct += 1

    accuracy = correct / total

    print("Accuracy: " + str(accuracy))
    '''
    plt.plot(history.history['accuracy'], label='accuracy')
    plt.plot(history.history['val_accuracy'], label='val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.xlim([1, 100])
    plt.ylim([0.1, 1.0])
    plt.legend(loc='lower right')
    plt.show()
    '''


run()