"""high level library for building neural networks in python with just a few lines of code."""

"""densely connected layer most basic kind of layer in neural network."""
"""____________________________simple NN for classification problems______________________"""
import keras
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from pathlib import Path

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
# x_train = actual images from the dataset
# y_train = matching label for each image
# x_test = additional images that we want to test to check the performance of NN
# y_test = y_test - This data has category labels for your test data, these labels will be used to test
# the accuracy between actual and predicted categories.

x_train = x_train.astype("float32")
x_test = x_test.astype("float32")
x_train = x_train / 255
x_test = x_test / 255

y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)


# defining our sequential neural network.

# model = keras.models.Sequential()
# adding a layer of three nodes and defining the network has two input nodes.

# model.add(Dense(3, input_dim=2))
# model.add(keras.Dense(3))
# rectified linear unit as an activation function(which input from the previous layer are enough to feed to the next layer.)

# model.add(Dense(3, activation='relu'))
# output layer

# model.add(Dense(1))

"""optimizer = algorithm used to train a neural network,
loss= loss function(how training proocess measures how right and wrong your neural network predictions.)"""

# model.compile(optimizer='adam',loss='mse')

""" for a small 256x256 pixel image
256x256x3 = 196,608 input nodes required.

___Translation invariance is an idea in which an ML model can recognize an object no matters whether it is moved 
in the image.So we add a new type of layer called convolution layer
(breaks apart the image in a way so that it can recognize same object in different positions)"""

"""1)break an image in to small overlapping tiles passing small window over the image and 
2) Now we pass each img tile through same neural network layer. tunning an img into array
each entry determine whether NN think of a certain pattern at that part of the image.

Max pooling: Down sample the databy only passing on the most important bits.
Divide the grid into 2x2 square and find the largest number in each square.

Dropout: A way to force a NN to try hard to learn instead of just memorize the training data.
Adding dropout layer btw other layers that will randomly throw away some of the data passing though it
by cutting some of connections in the NN."""












