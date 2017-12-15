#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
exercise of mnist with cnn

"""

import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data


mnist_data_sets = input_data.read_data_sets("MNIST_data/", one_hot=True)


learning_rate = 0.001
batch_size = 128
training_times = 10000
dropout = 0.5

train_x = mnist_data_sets.train.images
train_y = mnist_data_sets.train.labels
test_x = mnist_data_sets.train.images
test_y = mnist_data_sets.train.labels


# w = tf.Variable(tf.random_normal([784, 10]))
# b = tf.Variable(tf.constant(0.1))

# reshape to [-1, 28, 28, 1], -1 means ignore the quality of images,
# the size of image is 28 * 28, last 1 means the passage 1,
# because of the image has no color

train_x = train_x.reshape([-1, 28, 28, 1])
test_x = test_x.reshape([-1, 28, 28, 1])

x = tf.placeholder("float", [None, 28, 28, 1])
y_ = tf.placeholder("float", [None, 10])

# w is filter
w = tf.Variable(tf.random_normal(shape=[3, 3, 1, 32], stddev=0.01))
w1 = tf.Variable(tf.random_normal(shape=[3, 3, 32, 64], stddev=0.01))
w2 = tf.Variable(tf.random_normal(shape=[3, 3, 64, 128], stddev=0.01))
w3 = tf.Variable(tf.random_normal(shape=[128 * 4 * 4, 625], stddev=0.01))

w_o = tf.Variable(tf.random_normal(shape=[625, 10], stddev=0.01))

# first layer
# ksize in the max_pool is
# the size of the window for each dimension of the input tensor
layer_1 = tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding="SAME")
layer_1 = tf.nn.relu(layer_1)
layer_1_out = tf.nn.max_pool(layer_1, ksize=[1, 2, 2, 1],
                             strides=[1, 2, 2, 1],
                             padding="SAME")
layer_1_out = tf.nn.dropout(layer_1_out, keep_prob=dropout)

# 2nd layer
layer_2 = tf.nn.conv2d(layer_1_out, w1, strides=[1, 1, 1, 1], padding="SAME")
layer_2 = tf.nn.relu(layer_2)
layer_2_out = tf.nn.max_pool(layer_2, ksize=[1, 2, 2, 1],
                             strides=[1, 2, 2, 1],
                             padding="SAME")
layer_2_out = tf.nn.dropout(layer_2_out, keep_prob=dropout)

# 3rd layer
layer_3 = tf.nn.conv2d(layer_2_out, w2, strides=[1, 1, 1, 1], padding="SAME")
layer_3 = tf.nn.relu(layer_3)
layer_3_out = tf.nn.max_pool(layer_3, ksize=[1, 2, 2, 1],
                             strides=[1, 2, 2, 1],
                             padding="SAME")
# layer_3_out = layer_3_out.reshape([-1, w3.get_shape().as_list()[0]])
layer_3_out = tf.reshape(layer_3_out, [-1, 2048])
layer_3_out = tf.nn.dropout(layer_3_out, keep_prob=dropout)

# 4th layer
layer_4 = tf.nn.relu(tf.matmul(layer_3_out, w3))
layer_4_out = tf.nn.dropout(layer_4, 0.5)

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

step = 0
while step < training_times:
    print("training")
    accuracy_rate = sess.run(accuracy, feed_dict={x: test_x, y_: test_y})
    print(accuracy_rate)

    sess.run(train_step, feed_dict={x: train_x, y_: train_y})
    if step % 100 == 0:
        print("training")
    print("training")
    step += 1

    accuracy_rate = sess.run(accuracy, feed_dict={x: test_x, y_: test_y})
    print(accuracy_rate)
