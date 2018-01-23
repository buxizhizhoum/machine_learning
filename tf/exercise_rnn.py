#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
rnn only contains forward propagation, training not completed.
"""
import tensorflow as tf


X = [1.0, 2.0]  # input
# initial state
state = tf.Variable(tf.constant([[0.0, 0.0]], dtype=tf.float32))
# weights for state
w_state = tf.Variable(tf.constant([[0.1, 0.2], [0.3, 0.4]], dtype=tf.float32))
# weights for input
w_in = tf.Variable(tf.constant([[0.5, 0.6]], dtype=tf.float32))
# biases for input
b_in = tf.Variable(tf.constant([0.1, -0.1], dtype=tf.float32))

# weights for output
w_out = tf.Variable(tf.constant([[1.0], [2.0]], dtype=tf.float32))
# biases for output
b_out = tf.Variable(tf.constant(0.1, dtype=tf.float32))

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    for i in range(len(X)):
        # raw output before activation
        raw_out = tf.matmul(state, w_state) + tf.multiply(X[i], w_in) + b_in
        # update state, which will be used at next time
        state = tf.nn.tanh(raw_out)

        final_out = tf.matmul(state, w_out) + b_out

        print(sess.run(raw_out), sess.run(state), sess.run(final_out))






