#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
neural network, signal hidden layer
"""
import tensorflow as tf
import numpy as np

from tensorflow.examples.tutorials.mnist import input_data


learning_rate = 0.001
iter_num = 10
batch_size = 128

x = tf.placeholder("float", [None, 784])
y_ = tf.placeholder("float", [None, 10])
# w = tf.Variable(tf.random_normal([784, 10]))
w = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.random_uniform([10]))
# b = tf.Variable(tf.zeros([10]))


def model():
    # build model
    y = tf.add(tf.matmul(x, w), b)
    return y

y = model()
# cost function, cross entropy
cost = -tf.reduce_sum(y_ * tf.log(y))
train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
# accuracy of predict
accuracy = tf.reduce_mean(
    tf.cast(tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1)), "float"))
# initialize all variables
init = tf.global_variables_initializer()

if __name__ == "__main__":
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    test_x, test_y = mnist.test.images, mnist.test.labels

    with tf.Session() as sess:
        sess.run(init)

        step = 0
        while step < iter_num:
            for i in range(10000):
                train_x_batch, train_y_batch = mnist.train.next_batch(128)
                sess.run(train_step,
                         feed_dict={x: train_x_batch, y_: train_y_batch})

                accuracy_val = sess.run(accuracy,
                                        feed_dict={x: train_x_batch,
                                                   y_: train_y_batch})
                print(accuracy_val)
            step += 1
