"""VGG      16, 19  layers           2014
ResNet      50 layers                 2015
Inception v3
Mobile Net                             2017
NASNet                                end of 2017"""


import keras
from keras.datasets import cifar10
from keras.models import Sequential, model_from_json
from keras.preprocessing import image
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from keras.applications import vgg16


model = vgg16.VGG16()
img = image.load_img("bay.png", target_size=(224,224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = vgg16.preprocess_input(x)
predictions = model.predict(x)
predicted_classes = vgg16.decode_predictions(predictions, top=9)

print("Top predictions for this image:")


for imagenet_id, name, likelihood in predicted_classes[0]:
    print("Predictions: {} - {:2f"}).format(name, likelihood)

