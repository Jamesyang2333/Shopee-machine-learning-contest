'''This script goes along the blog post
"Building powerful image classification models using very little data"
from blog.keras.io.
It uses data that can be downloaded at:
https://www.kaggle.com/c/dogs-vs-cats/data
In our setup, we:
- created a data/ folder
- created train/ and validation/ subfolders inside data/
- created cats/ and dogs/ subfolders inside train/ and validation/
- put the cat pictures index 0-999 in data/train/cats
- put the cat pictures index 1000-1400 in data/validation/cats
- put the dogs pictures index 12500-13499 in data/train/dogs
- put the dog pictures index 13500-13900 in data/validation/dogs
So that we have 1000 training examples for each class, and 400 validation examples for each class.
In summary, this is our directory structure:
```
data/
    train/
        dogs/
            dog001.jpg
            dog002.jpg
            ...
        cats/
            cat001.jpg
            cat002.jpg
            ...
    validation/
        dogs/
            dog001.jpg
            dog002.jpg
            ...
        cats/
            cat001.jpg
            cat002.jpg
            ...
```
'''
import numpy as np
import cv2
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense
from keras import applications
from keras.preprocessing.image import img_to_array
from keras.optimizers import Adam

# dimensions of our images.
img_width, img_height = 224, 224
INIT_LR = 1e-3

top_model_weights_path = 'bottleneck_fc_model.h5'
train_data_dir = 'data/Training Images'
validation_data_dir = 'data/Training Images copy'
# nb_train_samples = 33894
# nb_validation_samples = 4317
nb_train_samples = 1000
nb_validation_samples = 289
epochs = 50
batch_size = 16


print([[1, 2]] * 2)
model = applications.VGG16(include_top=False, weights='imagenet')
image = cv2.imread("data/Training Images/BabyBibs/BabyBibs_2.jpg")
image = cv2.resize(image, (224, 224))
image = image.astype("float") / 255.0
image = img_to_array(image)
image = np.expand_dims(image, axis=0)
print(model.predict(image))
a = np.array([1, 2])
print(type(a))

