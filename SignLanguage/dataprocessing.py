from tensorflow import keras
import os
from tensorflow.keras.layers.experimental.preprocessing import RandomFlip
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tqdm import tqdm
import numpy as np


def generate_augmented_images(batch_size,
                              img_size,
                              normalize=True,
                              data_format="channels_last"):
    train_location = "dataset/train_images/"
    if (normalize == True):
        aug_gens = ImageDataGenerator(
            rescale=1.0 / 255,
            featurewise_center=False,
            samplewise_center=False,
            featurewise_std_normalization=False,
            samplewise_std_normalization=False,
            zca_whitening=False,
            validation_split=0.1,
            rotation_range=30,
            shear_range=0.25,
            zoom_range=0.3,
            width_shift_range=0.3,
            height_shift_range=0.3,
            horizontal_flip=False,
            vertical_flip=False,
            data_format=data_format,
        )
    else:
        aug_gens = ImageDataGenerator(
            featurewise_center=False,
            samplewise_center=False,
            featurewise_std_normalization=False,
            samplewise_std_normalization=False,
            zca_whitening=False,
            validation_split=0.1,
            rotation_range=30,
            shear_range=0.25,
            zoom_range=0.3,
            width_shift_range=0.3,
            height_shift_range=0.3,
            horizontal_flip=False,
            vertical_flip=False,
            data_format=data_format,
        )

    train_data = aug_gens.flow_from_directory(train_location,
                                              subset="training",
                                              seed=1447,
                                              target_size=(img_size, img_size),
                                              batch_size=batch_size,
                                              class_mode="categorical")

    val_data = aug_gens.flow_from_directory(train_location,
                                            subset="validation",
                                            seed=1447,
                                            target_size=(img_size, img_size),
                                            batch_size=batch_size,
                                            class_mode="categorical")

    return train_data, val_data


def generate_nonaugmented_images(batch_size, img_size, normalize=True):
    train_location = "dataset/train_images/"
    if (normalize == True):
        aug_gens = ImageDataGenerator(
            rescale=1.0 / 255,
            validation_split=0.1,
        )
    else:
        aug_gens = ImageDataGenerator(validation_split=0.1)

    train_data = aug_gens.flow_from_directory(train_location,
                                              subset="training",
                                              seed=1447,
                                              target_size=(img_size, img_size),
                                              batch_size=batch_size,
                                              class_mode="categorical")

    val_data = aug_gens.flow_from_directory(train_location,
                                            subset="validation",
                                            seed=1447,
                                            target_size=(img_size, img_size),
                                            batch_size=batch_size,
                                            class_mode="categorical")

    return train_data, val_data


def generate_test_labels():
    test_location = "dataset/test_images"
    img_id = []
    for fileName in os.listdir(test_location):
        img_id.append(fileName)
    return img_id


def tta_prediction(model,
                   batch_size,
                   img_size,
                   normalize=True,
                   data_format='channels_last'):
    test_location = "dataset/test_images"
    if (normalize == True):
        aug_gens = ImageDataGenerator(
            rescale=1.0 / 255,
            featurewise_center=False,
            samplewise_center=False,
            featurewise_std_normalization=False,
            samplewise_std_normalization=False,
            zca_whitening=False,
            rotation_range=10,
            shear_range=0.25,
            zoom_range=0.1,
            width_shift_range=0.1,
            height_shift_range=0.1,
            horizontal_flip=False,
            vertical_flip=False,
            data_format=data_format,
        )
    else:
        aug_gens = ImageDataGenerator(
            featurewise_center=False,
            samplewise_center=False,
            featurewise_std_normalization=False,
            samplewise_std_normalization=False,
            zca_whitening=False,
            rotation_range=10,
            shear_range=0.25,
            zoom_range=0.1,
            width_shift_range=0.1,
            height_shift_range=0.1,
            horizontal_flip=False,
            vertical_flip=False,
            data_format=data_format,
        )

    tta_steps = 10
    predictions = []
    for i in tqdm(range(tta_steps)):
        preds = model.predict(aug_gens.flow_from_directory(
            directory=test_location,
            target_size=(img_size, img_size),
            batch_size=batch_size,
            classes=['.'],
            shuffle=False,
        ),
                              workers=16)
        predictions.append(preds)

    final_pred = np.mean(predictions, axis=0)
    return final_pred


def get_data(batch_size, img_size):
    traindf = pd.read_csv("otherdataset/train/_annotations.csv", dtype=str)
    valdf = pd.read_csv("otherdataset/valid/_annotations.csv", dtype=str)
    testdf = pd.read_csv("otherdataset/test/_annotations.csv", dtype=str)

    aug_gens = ImageDataGenerator(
        featurewise_center=False,
        samplewise_center=False,
        featurewise_std_normalization=False,
        samplewise_std_normalization=False,
        zca_whitening=False,
        rotation_range=10,
        shear_range=0.25,
        zoom_range=0.1,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=False,
        vertical_flip=False,
        data_format='channels_first',
        fill_mode='nearest',
    )

    train = aug_gens.flow_from_dataframe(dataframe=traindf,
                                         directory="dataset/train/",
                                         x_col='filename',
                                         y_col='class',
                                         batch_size=batch_size,
                                         seed=1447,
                                         class_mode='categorical',
                                         target_size=(img_size, img_size))

    valid = aug_gens.flow_from_dataframe(dataframe=valdf,
                                         directory="dataset/valid/",
                                         x_col='filename',
                                         y_col='class',
                                         batch_size=batch_size,
                                         seed=1447,
                                         class_mode='categorical',
                                         target_size=(img_size, img_size))

    test = aug_gens.flow_from_dataframe(dataframe=testdf,
                                        directory="dataset/test/",
                                        x_col='filename',
                                        y_col='class',
                                        batch_size=batch_size,
                                        seed=1447,
                                        class_mode='categorical',
                                        target_size=(img_size, img_size))

    return train, valid, test
