#!/usr/bin/python
# -*- coding: utf-8 -*-
import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)


from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Activation


batch_size = 128
n_classes = 10
n_epoch = 12

img_rows, img_cols = 28, 28
n_filters = 32
pool_size = (2, 2)
kernel_size = (3, 3)

(X_train, y_train), (X_test, y_test) = mnist.load_data()

if K.image_dim_ordering() == "th":
    X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
    X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    # order of tf
    X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
    X_test = X_train.reshape(X_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)
X_train = X_train.astype("float32")
X_test = X_test.astype("float32")
X_train /= 255
X_test /= 255

# transform to matrix
Y_train = np_utils.to_categorial(y_train, n_classes)
Y_test = np_utils.to_categorial(y_test, n_classes)

# build model
model = Sequential()
model.add(Convolution2D)