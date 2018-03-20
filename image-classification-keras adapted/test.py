import csv
from keras.preprocessing.image import img_to_array
from keras import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.core import Activation, Flatten, Dense, Dropout
from imutils import paths
from keras import Model
from keras import applications
import numpy as np
import argparse
import imutils
import cv2
# Generate a model with all layers (with top)
vgg16 = applications.VGG16(weights=None, include_top=True)

#Add a layer where input is the output of the  second last layer
x = Dense(8, activation='softmax', name='predictions')(vgg16.layers[-2].output)

#Then create the corresponding model
my_model = Model(input=vgg16.input, output=x)
my_model.summary()