# ML Model for Recognizing Paddy Diseases

## Competition Link
https://www.kaggle.com/competitions/paddy-disease-classification/data

## Without Data Augmentation

### Basic Model
Mainly used for testing purposes.
Achieved 75% accuracy.

### MobileNet
Quick and Dirty Transfer learning by taking image and resizing it to 244x244 and running it through MobileNet

Achieved 84% accuracy.


### EfficientNetB0
Transfer Learning using pure EfficientNetB0 with top removed and using original image size.

Achieved 89% accuracy.

## With Data Augmentation

### Basic Model
Achieved 91% Accuracy

