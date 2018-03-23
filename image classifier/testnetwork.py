# USAGE
# python test_network.py --model santa_not_santa.model --image images/examples/santa_01.png

# import the necessary packages
import csv
from keras.preprocessing.image import img_to_array
from keras import Sequential
from keras.models import load_model
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.core import Activation, Flatten, Dense, Dropout
from imutils import paths
from keras import Model
from keras import applications
import numpy as np
import argparse
import imutils
import vgg
import cv2
lists = ["BabyBibs", "BabyHat", "BabyPants", "BabyShirt", "PackageFart", "womanshirtsleeve", "womencasualshoes", "womenchiffontop", "womendollshoes", "womenknittedtop", "womenlazyshoes", "womenlongsleevetop", "womenpeashoes", "womenplussizedtop", "womenpointedflatshoes", "womensleevelesstop", "womenstripedtop", "wrapsnslings"]

# construct the argument parse and parse the arguments
# ap = argparse.ArgumentParser()
# ap.add_argument("-m", "--model", required=True,
# 	help="path to trained model model")
# ap.add_argument("-i", "--image", required=True,
# 	help="path to input image")
# args = vars(ap.parse_args())

# load the image
# image = cv2.imread('Test/Test_3.jpg')
# orig = image.copy()
print("[INFO] loading network...")

model = load_model("resnet1.model")

answerlist = [0 for i in range(16111)]
imagepaths = list(paths.list_images('Test'))
# pre-process the image for classification
count = 0
for imagepath in imagepaths:
	print("{} images done".format(count))
	count = count + 1
	number = int(imagepath.split('.')[0][10:])
	image = cv2.imread(imagepath)
	image = cv2.resize(image, (224, 224))
	image = image.astype("float") / 255.0
	image = img_to_array(image)
	image = np.expand_dims(image, axis=0)
	problist = model.predict(image)[0]
	max = -1
	maxIndex = 0
	for i in range(18):
		if problist[i] > max:
			max = problist[i]
			maxIndex = i
	answerlist[number - 1] = maxIndex


printlist = [[str(i + 1), str(answerlist[i])] for i in range(16111)]

with open('result17.csv', 'w', newline='') as csvfile:
    spamwriter = csv.writer(csvfile)
    spamwriter.writerow(["id", "category"])
    spamwriter.writerows(printlist)

# load the trained convolutional neural network
# print("[INFO] loading network...")
# model = load_model('cloth.model')

# classify the input image
# problist = model.predict(image)[0]

# build the label
# label = "Santa" if santa > notSanta else "Not Santa"
# proba = santa if santa > notSanta else notSanta
# label = "{}: {:.2f}%".format(label, proba * 100)


# label = "{}: {:.2f}%".format(lists[maxIndex], max * 100)
# # draw the label on the image
# output = imutils.resize(orig, width=400)
# cv2.putText(output, label, (10, 25),  cv2.FONT_HERSHEY_SIMPLEX,
# 	0.7, (0, 255, 0), 2)
#
# # show the output image
# cv2.imshow("Output", output)
# cv2.waitKey(0)