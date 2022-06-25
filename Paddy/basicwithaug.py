import tensorflow as tf
import pandas as pd 
import matplotlib.pyplot as plt
from tensorflow.keras import regularizers
from tensorflow.keras.optimizers import SGD
from sklearn.model_selection import train_test_split
from tensorflow import keras
from keras.layers.advanced_activations import PReLU
import numpy as np
from PIL import Image
from tensorflow.keras import mixed_precision
import os
import csv
import tensorflow_hub as hub
import dataprocessing

policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)
tf.config.list_physical_devices("GPU")

### LABELS
label_arr = []
for i, line in enumerate(open("Dataset/wnids.txt", "r")):
    label_arr.append(line.rstrip("\n"))

batch_size = 300

### PARSING TRAIN/VALDIATION FILES
train_data, val_data = dataprocessing.generate_augmented_images(batch_size)

### PARSING TEST IMAGES
test_data = dataprocessing.generate_augmented_test(batch_size)
                                          

### Optimized Neural Network
model = keras.models.Sequential()

# Model Layers
model.add(
    keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu'))
model.add(keras.layers.MaxPooling2D((2, 2)))
model.add(
    keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu'))
model.add(keras.layers.MaxPooling2D((2, 2)))
model.add(
    keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
model.add(keras.layers.MaxPooling2D((2, 2)))
model.add(
    keras.layers.Conv2D(filters=128, kernel_size=(3, 3), activation='relu'))
model.add(keras.layers.MaxPooling2D((2, 2)))
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(1048, activation='swish'))
model.add(keras.layers.Dense(128, activation='swish'))
model.add(keras.layers.Dense(10, activation='softmax'))

model.build(input_shape=(None, 256, 256, 3))
model.summary()
model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss=tf.keras.losses.CategoricalCrossentropy(),
              metrics=['accuracy'])
num_epochs = 100
lr_reduction = keras.callbacks.ReduceLROnPlateau(monitor='val_loss',patience=4, verbose=1,  factor=0.4, min_lr=0.0001)

early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.00001, patience=8, mode='auto', restore_best_weights=True)

history = model.fit(train_data,
                    workers = 8,
                    epochs=num_epochs,
                    validation_data=val_data,
                    batch_size=batch_size,
                    verbose = 1,
                    callbacks = [early_stop, lr_reduction])

##Matching Predictions with Correct Image ID
predictions = model.evaluate(test_data, verbose = 1)
y_predict_max = np.argmax(model.predict(test_data),axis=1)

inverse_map = {v:k for k,v in train_data.class_indices.items()}

predictions = [inverse_map[k] for k in y_predict_max]

files=test_data.filenames

results=pd.DataFrame({"image_id":files,
                      "label":predictions})
results.image_id = results.image_id.str.replace('./', '')
results.to_csv("submission.csv",index=False)
results.head()

plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.1, 0.6])
plt.legend(loc='lower right')
plt.show()