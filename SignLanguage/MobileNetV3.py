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

keras.backend.set_image_data_format('channels_first')


def run():
    label_dict = {}
    for i, line in enumerate(open("Dataset/wnids.txt", "r")):
        label_dict[line.rstrip("\n")] = int(i)

    batch_size = 128
    img_size = 224
    num_classes = 26
    ### PARSING TRAIN/VALDIATION FILES
    train_data, val_data = dataprocessing.generate_augmented_images(
        batch_size, img_size, data_format="channels_first", normalize=False)

    ### PARSING TEST IMAGES
    test_labels = dataprocessing.generate_test_labels()
    test_int = [label_dict[x.replace(".jpg", "")] for x in test_labels]

    feature_extractor_layer = keras.applications.MobileNetV3Small(
        input_shape=(3, img_size, img_size),
        include_top=False,
        pooling='avg',
        include_preprocessing=True)

    ### Optimized Neural Network
    model = keras.models.Sequential()

    # Model Layers
    model.add(keras.Input(shape=(3, img_size, img_size)))
    model.add(feature_extractor_layer)
    model.add(keras.layers.Flatten())
    #model.add(keras.layers.Dense(1024, activation='swish'))
    model.add(keras.layers.Dense(256, activation='swish'))
    model.add(keras.layers.Dense(num_classes, activation='softmax'))

    model.build(input_shape=(None, 3, img_size, img_size))
    model.summary()
    model.compile(optimizer=tf.keras.optimizers.Adam(),
                  loss=tf.keras.losses.CategoricalCrossentropy(),
                  metrics=['accuracy'])

    num_epochs = 1
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
    model.save("ASLModel")

    ##Matching Predictions with Correct Image ID
    pred = dataprocessing.tta_prediction(model,
                                         batch_size,
                                         img_size,
                                         data_format="channels_first",
                                         normalize=False)
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

    # Convert the model
    converter = tf.lite.TFLiteConverter.from_saved_model(
        "ASLModel")  # path to the SavedModel directory
    tflite_model = converter.convert()

    # Save the model.
    with open('model.tflite', 'wb') as f:
        f.write(tflite_model)


run()