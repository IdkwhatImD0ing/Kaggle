from tensorflow import keras
import os


image_size = (480, 640)
batch_size = 8
def get_datasets():
    train_dataset = keras.utils.image_dataset_from_directory("Dataset/train_images", image_size = image_size, batch_size = batch_size, seed = 1447, validation_split = 0.2, subset = "training")
    val_dataset= keras.utils.image_dataset_from_directory("Dataset/train_images", image_size = image_size, batch_size = batch_size, seed = 1447, validation_split = 0.2, subset = "validation")
    return train_dataset, val_dataset

def get_test_dataset():
    img_id = []
    directory = "Dataset"
    testPath = directory + "/test_images/"
    for fileName in os.listdir(testPath):
        img_id.append(fileName)
    test_dataset = keras.utils.image_dataset_from_directory("Dataset/test_images", labels = None, image_size = image_size, batch_size = batch_size, shuffle = False)
    return test_dataset, img_id 


