#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
not passed
"""
import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

learning_rate = 0.001
training_iter = 100000
batch_size = 128  # todo: how to choose this value?

n_inputs = 28
n_steps = 28
n_hidden_units = 128  # units number in hidden layer
n_classes = 10


x = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
y = tf.placeholder(tf.float32, [None, n_classes])


weights = {
    "in": tf.Variable(tf.random_normal([n_inputs, n_hidden_units])),
    "out": tf.Variable(tf.random_normal([n_hidden_units, n_classes]))
}

biases = {
    "in": tf.Variable(tf.constant(0.1, shape=[n_hidden_units, ])),
    "out": tf.Variable(tf.constant(0.1, shape=[n_classes]))
}


def rnn(X, weights, biases):
    X = tf.reshape(X, [-1, n_inputs])
    x_in = tf.matmul(X, weights["in"]) + biases["in"]
    x_in = tf.reshape(x_in, [-1, n_steps, n_hidden_units])

    # lstm_cell = tf.contrib.rnn.BasicLSTMCell(n_hidden_units,
    #                                          forget_bias=1,
    #                                          state_is_tuple=True)
    lstm_cell = tf.contrib.rnn.BasicLSTMCell(n_hidden_units, forget_bias=1.0,
                                        state_is_tuple = True)
    init_state = lstm_cell.zero_state(batch_size, dtype=tf.float32)
    # output, final_state = tf.nn.dynamic_rnn(lstm_cell, x_in,
    #                                         initial_state=init_state,
    #                                         time_major=False)
    outputs, final_state = tf.nn.dynamic_rnn(lstm_cell, x_in,
                                             initial_state=init_state,
                                             time_major=False)
    results = tf.matmul(final_state[1], weights["out"]) + biases["out"]
    return results


if __name__ == "__main__":
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

    train_x, train_y = mnist.train.images, mnist.train.labels
    test_x, test_y = mnist.test.images, mnist.test.labels

    predict = rnn(train_x, weights, biases)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
        logits=predict, labels=train_y))
    train_op = tf.train.AdamOptimizer(learning_rate).minimize(cost)

    predict_accuracy = tf.equal(tf.arg_max(predict, 1), tf.arg_max(train_y, 1))
    accuracy = tf.reduce_mean(tf.cast(predict_accuracy, tf.float32))

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        step = 0
        while step * batch_size < training_iter:
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            batch_xs = batch_xs.Reshape([batch_size, n_steps, n_inputs])
            sess.run(train_op, feed_dict={x: batch_xs, y: batch_ys})

            if step % 20 == 0:
                print(sess.run(accuracy, feed_dict={x: batch_xs, y: batch_ys}))
                step += 1

