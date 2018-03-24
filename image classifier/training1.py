from keras import applications
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from imutils import paths
import cv2
import numpy as np
from keras.preprocessing.image import img_to_array
import vgg
import random
import os

# path to the model weights files.
weights_path = 'vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'
top_model_weights_path = 'bottleneck_fc_model1.h5'
# dimensions of our images.
img_width, img_height = 224, 224

train_data_dir = 'data/Training Images'
validation_data_dir = 'data/Training Images copy'
nb_train_samples = 33894
nb_validation_samples = 4317
epochs = 50
batch_size = 32

# build the VGG16 network
model = vgg.VGG_16()
model.load_weights("vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5")
print('Model loaded.')

# build a classifier model to put on top of the convolutional model
top_model = Sequential()
top_model.add(Flatten(input_shape=(7, 7, 512)))
top_model.add(Dense(4096, activation='relu'))
top_model.add(Dropout(0.6))
top_model.add(Dense(4096, activation='relu'))
top_model.add(Dropout(0.7))
top_model.add(Dense(18, activation='softmax'))
top_model.load_weights(top_model_weights_path)

model.add(top_model)

# set the first 25 layers (up to the last conv block)
# to non-trainable (weights will not be updated)
for layer in model.layers[:11]:
    layer.trainable = False

# compile the model with a SGD/momentum optimizer
# and a very slow learning rate.
model.compile(loss='categorical_crossentropy',
              optimizer=optimizers.SGD(lr=1e-3, momentum=0.9),
              metrics=['accuracy'])
# initialize the number of epochs to train for, initia learning rate,
# and batch size
EPOCHS = 30
BS = 32
lists = ["BabyBibs", "BabyHat", "BabyPants", "BabyShirt", "PackageFart", "womanshirtsleeve", "womencasualshoes", "womenchiffontop", "womendollshoes", "womenknittedtop", "womenlazyshoes", "womenlongsleevetop", "womenpeashoes", "womenplussizedtop", "womenpointedflatshoes", "womensleevelesstop", "womenstripedtop", "wrapsnslings"]


# initialize the data and labels
print("[INFO] loading images...")
data = []
labels = []

# grab the image paths and randomly shuffle them
imagePaths = list(paths.list_images("data/Training Images"))
random.seed(42)
random.shuffle(imagePaths)

# loop over the input images
for imagePath in imagePaths:
	# load the image, pre-process it, and store it in the data list
	image = cv2.imread(imagePath)
	image = cv2.resize(image, (224, 224))
	image = img_to_array(image)
	data.append(image)

	# extract the class label from the image path and update the
	# labels list
	label = imagePath.split(os.path.sep)[-2]
	label = lists.index(label)
	labels.append(label)

# scale the raw pixel intensities to the range [0, 1]
data = np.array(data, dtype="float") / 255.0
labels = np.array(labels)

# partition the data into training and testing splits using 75% of
# the data for training and the remaining 25% for testing
(trainX, testX, trainY, testY) = train_test_split(data,
	labels, test_size=0.25, random_state=42)

# convert the labels from integers to vectors
trainY = to_categorical(trainY, num_classes=18)
testY = to_categorical(testY, num_classes=18)

# construct the image generator for data augmentation
aug = ImageDataGenerator(rotation_range=30, width_shift_range=0.1,
	height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
	horizontal_flip=True, fill_mode="nearest")

# initialize the mod

# train the network
print("[INFO] training network...")
H = model.fit_generator(aug.flow(trainX, trainY, batch_size=BS),
	validation_data=(testX, testY), steps_per_epoch=len(trainX) // BS,
	epochs=EPOCHS, verbose=1)

