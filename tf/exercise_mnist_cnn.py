#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
exercise of mnist with cnn

ref: mnist_from_web.py from https://github.com/nlintz

summary:
    1. should not pass too many data to model at one time, when the all train
    data set is passed as input in one time, the memory is exhausted, same
    situation happened when the all test data set is passed in one time.
    the memory is limited, if the memory is exhausted, there will be error:
    Process finished with exit code 137 (interrupted by signal 9: SIGKILL)

    2.reshape could not be done, after the data sets have been get
    with next_batch(), still need to know why?


todo: add bias?
"""

import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.python import debug as tf_debug


# data_path = "/home/think/2_study/1_machine_learning/machine_learning/tf/MNIST_data"
# mnist_data_sets = input_data.read_data_sets(data_path, one_hot=True)
# if the mnist data set is located at same folder with this script,
# use code below
mnist_data_sets = input_data.read_data_sets("MNIST_data/", one_hot=True)

learning_rate = 0.001
batch_size = 128
training_times = 100
# dropout = 0.5
# dropout = tf.Variable(tf.constant(0.5), "float")
dropout = tf.placeholder("float")

train_x = mnist_data_sets.train.images
train_y = mnist_data_sets.train.labels
test_x = mnist_data_sets.train.images
test_y = mnist_data_sets.train.labels


# reshape to [-1, 28, 28, 1], -1 means ignore the quality of images,
# the size of image is 28 * 28, last 1 means the passage 1,
# because of the image has no color

# todo: what is the difference between the tow reshape below [] or not?
train_x = train_x.reshape([-1, 28, 28, 1])
test_x = test_x.reshape([-1, 28, 28, 1])

# train_x = train_x.reshape(-1, 28, 28, 1)
# test_x = test_x.reshape(-1, 28, 28, 1)

# not work
# train_x = tf.reshape(train_x, [-1, 28, 28, 1])
# test_x = tf.reshape(test_x, [-1, 28, 28, 1])

x = tf.placeholder("float", [None, 28, 28, 1])
y_ = tf.placeholder("float", [None, 10])

# w is filter
w = tf.Variable(tf.random_normal(shape=[3, 3, 1, 32], stddev=0.01))
w2 = tf.Variable(tf.random_normal(shape=[3, 3, 32, 64], stddev=0.01))
w3 = tf.Variable(tf.random_normal(shape=[3, 3, 64, 128], stddev=0.01))
w4 = tf.Variable(tf.random_normal(shape=[128 * 4 * 4, 625], stddev=0.01))

w_o = tf.Variable(tf.random_normal(shape=[625, 10], stddev=0.01))

# first layer
# ksize in the max_pool is
# the size of the window for each dimension of the input tensor
layer_1 = tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding="SAME")
layer_1 = tf.nn.relu(layer_1)
layer_1_out = tf.nn.max_pool(layer_1, ksize=[1, 2, 2, 1],
                             strides=[1, 2, 2, 1],
                             padding="SAME")
layer_1_out = tf.nn.dropout(layer_1_out, dropout)

# 2nd layer
layer_2 = tf.nn.conv2d(layer_1_out, w2, strides=[1, 1, 1, 1], padding="SAME")
layer_2 = tf.nn.relu(layer_2)
layer_2_out = tf.nn.max_pool(layer_2, ksize=[1, 2, 2, 1],
                             strides=[1, 2, 2, 1],
                             padding="SAME")
layer_2_out = tf.nn.dropout(layer_2_out, dropout)

# 3rd layer
layer_3 = tf.nn.conv2d(layer_2_out, w3, strides=[1, 1, 1, 1], padding="SAME")
layer_3 = tf.nn.relu(layer_3)
layer_3_out = tf.nn.max_pool(layer_3, ksize=[1, 2, 2, 1],
                             strides=[1, 2, 2, 1],
                             padding="SAME")
# layer_3_out = tf.reshape(layer_3_out, [-1, w4.get_shape().as_list()[0]])
# reshape to a vector in order to used as the input of fc layer.
layer_3_out = tf.reshape(layer_3_out, [-1, 2048])
layer_3_out = tf.nn.dropout(layer_3_out, dropout)

# 4th layer
layer_4 = tf.nn.relu(tf.matmul(layer_3_out, w4))
layer_4_out = tf.nn.dropout(layer_4, 0.5)
# layer_4_out = layer_4

# full connected layer
res = tf.matmul(layer_4_out, w_o)
# does softmax needed?


# cost function
cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits_v2(logits=res, labels=y_))

train_step = tf.train.RMSPropOptimizer(learning_rate).minimize(cross_entropy)

# accuracy
predict_accuracy = tf.equal(tf.argmax(res, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(predict_accuracy, "float"))


# initialize
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
# debug
# sess = tf_debug.LocalCLIDebugWrapperSession(sess, ui_type="readline")

step = 0
while step < training_times:
    # should not pass too many data at one time, the memory is limited!
    index_start = range(0, len(train_x), batch_size)
    # index_end = range(batch_size, len(train_x)+1, batch_size)
    # index_dual = zip(index_start, index_end)
    # for start, end in index_dual:
    for start in index_start:
        end = start + batch_size
        train_x_batch = train_x[start: end]
        train_y_batch = train_y[start: end]

        test_x_batch = test_x[start: end]
        test_y_batch = test_y[start: end]
        sess.run(train_step,
                 feed_dict={x: train_x_batch,
                            y_: train_y_batch,
                            dropout: 0.5})

        print("training")
        # should not pass too many data at one time, the memory is limited!
        accuracy_rate = sess.run(accuracy,
                                 feed_dict={x: test_x_batch,
                                            y_: test_y_batch,
                                            dropout: 0.5})
        print("accuracy:", accuracy_rate)
    step += 1


# while step < training_times:
#     # should not pass too many data at one time, the memory is limited!
#     # index_start = range(0, len(train_x), batch_size)
#     # index_end = range(batch_size, len(train_x)+1, batch_size)
#     # index_dual = zip(index_start, index_end)
#     # for start, end in index_dual:
#     # for start in index_start:
#     #     end = start + batch_size
#     #     train_x_batch = train_x[start: end]
#     #     train_y_batch = train_y[start: end]
#     #
#     #     test_x_batch = test_x[start: end]
#     #     test_y_batch = test_y[start: end]
#     train_x_batch, train_y_batch = mnist_data_sets.train.next_batch(batch_size)
#     test_x_batch, test_y_batch = mnist_data_sets.train.next_batch(batch_size)
#     # train_x_batch = tf.reshape(train_x_batch, [-1, 28, 28, 1])
#     # test_x_batch = tf.reshape(test_x_batch, [-1, 28, 28, 1])
#     sess.run(train_step,
#              feed_dict={x: train_x_batch,
#                         y_: train_y_batch,
#                         dropout: 0.5})
#
#     print("training")
#     # should not pass too many data at one time, the memory is limited!
#     accuracy_rate = sess.run(accuracy,
#                              feed_dict={x: test_x_batch,
#                                         y_: test_y_batch,
#                                         dropout: 0.5})
#     print("accuracy:", accuracy_rate)
#     step += 1


sess.close()
