import tensorflow as tf
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

tf.keras.mixed_precision.experimental.set_policy('mixed_float16')
batch_size = 128
### LABELS
label_arr = []
for i, line in enumerate(open("Dataset/wnids.txt", "r")):
    label_arr.append(line.rstrip("\n"))


### PARSING TRAIN/VALDIATION FILES
train_dataset, val_dataset = dataprocessing.get_datasets(batch_size)

### PARSING TEST IMAGES
test_dataset, img_id = dataprocessing.get_test_dataset(batch_size)
                                          

effModel = keras.applications.EfficientNetB0(weights='imagenet',
                                             pooling = 'avg',
                                             include_top=False,
                                             input_shape=(224, 224, 3))
effModel.trainable = False
### Optimized Neural Network
model = keras.models.Sequential()
# Data Augmentation
#model.add(keras.layers.RandomFlip("horizontal_and_vertical"))
#model.add(keras.layers.RandomRotation(0.2))
#model.add(keras.layers.RandomContrast(0.2))
#model.add(keras.layers.RandomZoom(0.2))

# Model Layers
model.add(keras.layers.Resizing(224, 224))
model.add(effModel)
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(256, activation='swish'))
model.add(keras.layers.Dropout(0.4))
model.add(keras.layers.Dense(256, activation='swish'))
model.add(keras.layers.Dropout(0.4))
model.add(keras.layers.Dense(10, activation='softmax'))

model.build(input_shape=(None, 480, 640, 3))
model.summary()
model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
              metrics=['accuracy'])

history = model.fit(train_dataset,
                    epochs=20,
                    validation_data=val_dataset,
                    batch_size = batch_size)

predictions = model.predict(test_dataset, batch_size=batch_size)
predictions = np.argmax(predictions, axis=1)

### Matching Predictions with correct image id
y_test = []
for i in range(len(predictions)):
    y_test.append(label_arr[predictions[i]])


combined = [[i, j] for i, j in zip(img_id, y_test)]
with open("predictions.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerows([["image_id", "label"]])
    writer.writerows(combined)

plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.1, 0.6])
plt.legend(loc='lower right')
plt.show()