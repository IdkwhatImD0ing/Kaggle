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

### LABELS
label_arr = []
for i, line in enumerate(open("Dataset/wnids.txt", "r")):
    label_arr.append(line.rstrip("\n"))


### PARSING TRAIN/VALDIATION FILES
image_size = (480, 640)
batch_size = 16
train_dataset = keras.utils.image_dataset_from_directory("Dataset/train_images", image_size = image_size, batch_size = batch_size, seed = 1447, validation_split = 0.2, subset = "training")
val_dataset= keras.utils.image_dataset_from_directory("Dataset/train_images", image_size = image_size, batch_size = batch_size, seed = 1447, validation_split = 0.2, subset = "validation") 
print("Finished Parsing")


### PARSING TEST IMAGES
img_id = []
directory = "Dataset"
testPath = directory + "/test_images/"
for fileName in os.listdir(testPath):
    img_id.append(fileName)
test_dataset = keras.utils.image_dataset_from_directory("Dataset/test_images", labels = None, image_size = image_size, batch_size = batch_size, shuffle = False)
print("Finished Converting")
                                          

### Optimized Neural Network
model = keras.models.Sequential()
# Data Augmentation
#model.add(keras.layers.RandomFlip("horizontal_and_vertical"))
#model.add(keras.layers.RandomRotation(0.2))
#model.add(keras.layers.RandomContrast(0.2))
#model.add(keras.layers.RandomZoom(0.2))

# Model Layers
model.add(keras.layers.Rescaling(scale = 1./255))
model.add(
    keras.layers.Conv2D(filters=16, kernel_size=(3, 3), activation='relu'))
model.add(
    keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
model.add(keras.layers.MaxPooling2D((2, 2)))
model.add(
    keras.layers.Conv2D(filters=128, kernel_size=(3, 3), activation='relu'))
model.add(keras.layers.MaxPooling2D((2, 2)))
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(10, activation='softmax'))

model.build(input_shape=(None, 480, 640, 3))
model.summary()
model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
              metrics=['accuracy'])

history = model.fit(train_dataset,
                    epochs=5,
                    validation_data=val_dataset,
                    batch_size=16)

predictions = model.predict(test_dataset, batch_size=32)
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