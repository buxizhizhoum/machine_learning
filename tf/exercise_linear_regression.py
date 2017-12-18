#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
linear regression

if the learning rate is too large, the optimized k will increase crazily
and finally the k will be inf or nan
"""

import tensorflow as tf
import numpy as np


x = tf.placeholder("float")
y = tf.placeholder("float")
k = tf.Variable(0.0, "float")
b = tf.Variable(0.1, "float")

learning_rate = 0.0001


def build_model(xx, kk, b=None):
    # model = k * x + b
    return tf.add(tf.multiply(kk, xx), b)

model = build_model(x, k, b)

cost = tf.square(y - model)
# cost = -tf.reduce_sum(y * tf.log(model))  # not work , k is always increase?
train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

init = tf.global_variables_initializer()
# abs(y - y_)
delta_y = tf.reduce_mean(tf.abs(tf.subtract(y, model)))

if __name__ == "__main__":
    # create test data
    test_x = np.linspace(-1, 1, 1000)
    test_y = 3 * test_x + 50.0 + np.random.rand(*test_x.shape)

    with tf.Session() as sess:
        sess.run(init)
        for i in range(1000):
            # print(sess.run(k))
            print(sess.run(delta_y, feed_dict={x: test_x, y: test_y}))
            sess.run(train_step, feed_dict={x: test_x, y: test_y})
        print("k is :", sess.run(k))
        print("b is :", sess.run(b))
