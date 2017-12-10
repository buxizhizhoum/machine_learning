#!/usr/bin/python
# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np


def add_layer(inputs, in_size, out_size, active_function=None):
    weights = tf.Variable(tf.random_normal([in_size, out_size]))
    biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)
    # f(x) = Wx + b
    func = tf.matmul(inputs, weights) + biases
    if active_function is None:
        outputs = func
    else:
        outputs = active_function(func)

    return outputs






if __name__ == "__main__":
    x_data = np.linspace(-1, 1, 300)[:, np.newaxis]
    noise = np.random.normal(0, 0.05, x_data.shape)
    y_data = np.square(x_data) - 0.5 + noise  # y = x^2 -0.5
    xs = tf.placeholder(tf.float32, [None, 1])
    ys = tf.placeholder(tf.float32, [None, 1])

    h_1 = add_layer(xs, 1, 20, active_function=tf.nn.relu)
    prediction = add_layer(h_1, 20, 1, active_function=None)

    loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys-prediction),
                          reduction_indices=[1]))
    train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

    # loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction),
    #                       reduction_indices=[1]))
    # train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

    init = tf.global_variables_initializer()
    sess = tf.Session()

    sess.run(init)

    for i in range(1000):
        sess.run(train_step, feed_dict={xs: x_data, ys: y_data})
        if i % 100 == 0:
            print(sess.run(loss, feed_dict={xs: x_data, ys: y_data}))


