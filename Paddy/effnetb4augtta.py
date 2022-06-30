import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd 
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

batch_size = 16
img_size = 380
### PARSING TRAIN/VALDIATION FILES
train_data, val_data = dataprocessing.generate_augmented_images(batch_size, img_size, normalize = False)

### PARSING TEST IMAGES
test_data = dataprocessing.generate_augmented_test(batch_size, img_size, normalize = False)
                                          
effModel = keras.applications.EfficientNetB4(weights='imagenet',
                                             pooling = 'avg',
                                             include_top=False,
                                             input_shape=(img_size, img_size, 3))

### Optimized Neural Network
model = keras.models.Sequential()

# Model Layers
model.add(effModel)
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(1024, activation='swish'))
model.add(keras.layers.Dense(512, activation='swish'))
model.add(keras.layers.Dense(256, activation='swish'))
model.add(keras.layers.Dense(10, activation='softmax'))

model.build(input_shape=(None, img_size, img_size, 3))
model.summary()
model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss=tf.keras.losses.CategoricalCrossentropy(),
              metrics=['accuracy'])

num_epochs = 100
lr_reduction = keras.callbacks.ReduceLROnPlateau(monitor='val_loss',patience=4, verbose=1,  factor=0.4, min_lr=0.0001)

early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.00001, patience=8, mode='auto', restore_best_weights=True)

history = model.fit(train_data,
                    workers = 16,
                    epochs=num_epochs,
                    validation_data=val_data,
                    batch_size=batch_size,
                    verbose = 1,
                    callbacks = [early_stop, lr_reduction])

##Matching Predictions with Correct Image ID
pred = dataprocessing.tta_prediction(model, batch_size, img_size, normalize = False)
y_predict_max = np.argmax(pred, axis = 1)

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