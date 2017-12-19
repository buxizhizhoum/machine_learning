#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
logistic regression, the result is wrong, need to check...
"""
import tensorflow as tf
import numpy as np


x = tf.placeholder("float")
y_ = tf.placeholder("float")

w = tf.Variable(0.0, "float")
b = tf.Variable(0.1, "float")


def build_model():
    tmp = tf.add(tf.multiply(x, w), b)
    return tf.sigmoid(tmp)

y = build_model()

cost = -tf.reduce_sum(y_ * tf.log(y))
train_step = tf.train.GradientDescentOptimizer(0.0001).minimize(cost)

init = tf.global_variables_initializer()


if __name__ == "__main__":
    test_x = np.linspace(-1, 1, 1000)
    test_y = test_x

    with tf.Session() as sess:
        sess.run(init)
        for i in range(1000):
            print(sess.run(w))
            sess.run(train_step, feed_dict={x: test_x, y_: test_y})
        print("result")
        print(sess.run(w))
        print(sess.run(b))
