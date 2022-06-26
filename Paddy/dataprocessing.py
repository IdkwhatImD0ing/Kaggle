from tensorflow import keras
import os
from tensorflow.keras.layers.experimental.preprocessing import RandomFlip
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.preprocessing.image import ImageDataGenerator

image_size = (480, 640)


def get_datasets(batch_size):
    train_dataset = keras.utils.image_dataset_from_directory(
        "Dataset/train_images",
        image_size=image_size,
        batch_size=batch_size,
        seed=1447,
        validation_split=0.2,
        subset="training")
    val_dataset = keras.utils.image_dataset_from_directory(
        "Dataset/train_images",
        image_size=image_size,
        batch_size=batch_size,
        seed=1447,
        validation_split=0.2,
        subset="validation")
    return train_dataset, val_dataset


def get_test_dataset(batch_size):
    img_id = []
    directory = "Dataset"
    testPath = directory + "/test_images/"
    for fileName in os.listdir(testPath):
        img_id.append(fileName)
    test_dataset = keras.utils.image_dataset_from_directory(
        "Dataset/test_images",
        labels=None,
        image_size=image_size,
        batch_size=batch_size,
        shuffle=False)
    return test_dataset, img_id


def generate_augmented_images(batch_size, img_size, normalize = True):
    train_location = "Dataset/train_images/"
    if(normalize == True):
        aug_gens = ImageDataGenerator(
            rescale = 1.0/255,
            featurewise_center=False,
            samplewise_center=False,
            featurewise_std_normalization=False,
            samplewise_std_normalization=False,
            zca_whitening=False,
            validation_split=0.1,
            rotation_range=10,
            shear_range=0.25,
            zoom_range=0.1,
            width_shift_range=0.1,
            height_shift_range=0.1,
            horizontal_flip=True,
            vertical_flip=True,
        )
    else:
        aug_gens = ImageDataGenerator(
            featurewise_center=False,
            samplewise_center=False,
            featurewise_std_normalization=False,
            samplewise_std_normalization=False,
            zca_whitening=False,
            validation_split=0.1,
            rotation_range=10,
            shear_range=0.25,
            zoom_range=0.1,
            width_shift_range=0.1,
            height_shift_range=0.1,
            horizontal_flip=True,
            vertical_flip=True,
        )

    train_data = aug_gens.flow_from_directory(train_location,
                                              subset="training",
                                              seed=1447,
                                              target_size=(img_size,img_size),
                                              batch_size=batch_size,
                                              class_mode="categorical")

    val_data = aug_gens.flow_from_directory(train_location,
                                            subset="validation",
                                            seed=1447,
                                            target_size=(img_size,img_size),
                                            batch_size=batch_size,
                                            class_mode="categorical")

    return train_data, val_data

def generate_augmented_test(batch_size, img_size, normalize = True):
    test_location = "Dataset/test_images"
    if(normalize == True):
        test_data = ImageDataGenerator(rescale=1.0/255).flow_from_directory(    
            directory=test_location,
            target_size=(img_size, img_size),
            batch_size=batch_size,
            classes=['.'],
            shuffle=False,
        )
    else:
        test_data = ImageDataGenerator().flow_from_directory(    
            directory=test_location,
            target_size=(img_size, img_size),
            batch_size=batch_size,
            classes=['.'],
            shuffle=False,
        )
    return test_data

def tta_prediction(model, batch_size, img_size, normalize = True):
    test_location = "Dataset/test_images"
    if(normailize == True):
        aug_gens = ImageDataGenerator(
            rescale = 1.0/255,
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
            horizontal_flip=True,
            vertical_flip=True,
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
            horizontal_flip=True,
            vertical_flip=True,
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
        ))
        predictions.append(preds)

    final_pred = np.mean(predictions, axis=0)
    return final_pred