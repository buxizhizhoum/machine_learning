#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
cnn which is used to solve MNIST problem, slim is used to simplify program.
"""

import tensorflow as tf
import tensorflow.contrib.slim as slim

from tensorflow.examples.tutorials.mnist import input_data


LEARNING_RATE = 0.01
TRAINING_STEPS = 10000
BATCH_SIZE = 100


def model(x):
    # tf.reshape(x, [None, 28, 28])
    # x = tf.reshape(x, [-1, 28, 28, 1])
    layer_1 = slim.conv2d(x, 6, [3, 3])
    pool_1 = slim.max_pool2d(layer_1, [2, 2])

    layer_2 = slim.conv2d(pool_1, 16, [5, 5], padding="VALID")
    pool_2 = slim.max_pool2d(layer_2, [2, 2])
    # why reshape
    pool_2 = tf.reshape(pool_2, [-1, 400])

    fc_1 = slim.fully_connected(pool_2, 120)
    fc_2 = slim.fully_connected(fc_1, 84)
    res = slim.fully_connected(fc_2, 10, activation_fn=None)

    # res = slim.softmax(fc_3)

    return res


def train(mnist):
    x = tf.placeholder(tf.float32, [None, 28, 28, 1])
    y_ = tf.placeholder(tf.float32, [None, 10])

    y = model(x)

    # loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
    #     logits=y, labels=y_))

    loss = slim.losses.softmax_cross_entropy(logits=y, onehot_labels=y_)
    train_step = tf.train.GradientDescentOptimizer(
        LEARNING_RATE).minimize(loss)

    correct_predict = tf.equal(tf.arg_max(y, 1), tf.arg_max(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_predict, tf.float32))

    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)

        for i in range(TRAINING_STEPS*10):
            x_batch, y_batch = mnist.train.next_batch(100)
            x_batch = x_batch.reshape([-1, 28, 28, 1])  # attention here

            sess.run(train_step, feed_dict={x: x_batch, y_: y_batch})

            if i % 100 == 0:
                accuracy_rate = sess.run(
                    accuracy,
                    feed_dict={x: x_batch, y_: y_batch})
                print("accuracy: %s" % accuracy_rate)


def main(argv=None):
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    train(mnist)


if __name__ == "__main__":
    tf.app.run()
