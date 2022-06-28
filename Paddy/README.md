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

### MobileNet
Only achieved 87% accuracy, but still impressive for something that has 1/6 the parameters as the basic model.

### EfficientNetB0
Achieved 91% Accuracy.

### EfficientNetB2
Achieved 93% Accuracy

### EfficientNetB4
Achieved 94.7% Accuracy

### EfficientNetB7
Achieved 96.1% Accuracy


## Using Test Time Augmentation

## Mobile Net
Achieved staggering 96.3% Accuracy, same as EfficientNetB7 but in around 1/20 the time.

## EfficientNetB0
Achieved 97.30% Accuracy

## EfficientNetB7
Achieved 97.46% Accuracy

## ConvNeXtTiny
Achieved 97.00% Accuracy