# ASL Recognition

ML for recognizing ASL for SNAP AR hackathon

## Dataset

Used dataset from following link with del and nothing removed since not useful for our usage.
https://www.kaggle.com/datasets/grassknoted/asl-alphabet

## Machine Learning Models:

Both Validation Accuracy and Test Accuracy is included, but due to small size of test dataset, Validation Accuracy is a better indicator of model performance.

### Basic:

Handmade model with image augmentation

Short Training Time

Validation Accuracy: 95.04%

Test Accuracy using TTA: 100%

### MobileNet:

Not using due to model being to big to fit in Lens Studio

Transfer learning without Image Augmentation

Short Training Time

Validation Accuracy: 91.41%

Test Accuracy using TTA: 100%

### MobileNetV3:

Not using due to model unable to import into Lens Studio

Transfer learning without Image Augmentation

Short Training Time

Validation Accuracy: 99.41%

Test Accuracy using TTA: 100%

### EfficientNetB0:

Not using due to unable to save model

Transfer learning model using Image Augmentation

Long training time

Validation Accuracy: 99.83%

Test Accuracy using TTA: 100%
