#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
rnn, to train and test with mnist data sets.
"""

import tensorflow as tf
import numpy as np

from tensorflow.examples.tutorials.mnist import input_data


l_rate = 0.01
batch_size = 128
iter_num = 30


x = tf.placeholder("float", [None, 28, 28])
y_ = tf.placeholder("float", [None, 10])


w_in = tf.Variable(tf.random_normal([28, 128]))
w_out = tf.Variable(tf.random_normal([128, 10]))

b_in = tf.Variable(tf.random_normal([128]))
b_out = tf.Variable(tf.random_normal([10]))


def model(x, w_in, w_out, b_in, b_out):
    x_in = tf.reshape(x, [-1, 28])
    y_in_o = tf.add(tf.matmul(x_in, w_in), b_in)
    # input of next layer
    x_out_in = tf.reshape(y_in_o, [-1, 28, 128])
    # rnn
    lstm_c = tf.nn.rnn_cell.BasicLSTMCell(128,
                                          forget_bias=1.0,
                                          state_is_tuple=True)
    init = lstm_c.zero_state(batch_size, dtype="float")
    output, finally_state = tf.nn.dynamic_rnn(
        lstm_c, x_out_in, initial_state=init, time_major=False)
    result = tf.add(tf.matmul(finally_state[1], w_out), b_out)
    return result

y = model(x, w_in, w_out, b_in, b_out)
cost = tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_, logits=y)
train_step = tf.train.AdamOptimizer(l_rate).minimize(cost)

accuracy_list = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(accuracy_list, "float"))

init = tf.global_variables_initializer()

if __name__ == "__main__":
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    test_x, test_y = mnist.test.images, mnist.test.labels
    # reshape test_x
    test_x = test_x.reshape([-1, 28, 28])

    with tf.Session() as sess:
        # initialize all variables
        sess.run(init)

        step = 0
        while step < iter_num:
            # for i in range(1000):
            #     train_x_batch, train_y_batch \
            #         = mnist.train.next_batch(batch_size)
            #     # reshape train data
            #     train_x_batch = train_x_batch.reshape([batch_size, 28, 28])
            #     sess.run(train_step,
            #              feed_dict={x: train_x_batch, y_: train_y_batch})
            #
            #     if i % 100 == 0:
            #         # calculate accuracy
            #         accuracy_train = sess.run(accuracy,
            #                                   feed_dict={x: train_x_batch,
            #                                              y_: train_y_batch})
            #         print("train accuracy: %s" % accuracy_train)

            step += 1
            # todo: not very clear here
            # get a slice of test_x
            test_x_index = np.arange(len(test_x))
            # random sort the test_x_index, this will be used
            # to get data from test_x
            np.random.shuffle(test_x_index)
            # get a slice from text_x_index, it is random index,
            # so the finally test_x data set get with those index is random
            test_batch_index = test_x_index[0:128]  # size is 128
            # get random test data from test_x
            # this method only support in numpy, to get some item from a array
            # with a index in type of list
            test_x_batch = test_x[test_batch_index]  # numpy array support
            test_y_batch = test_y[test_batch_index]

            # both of numpy and python support method below
            # test_x_batch = [test_x[i] for i in test_batch_index]
            # test_y_batch = [test_y[i] for i in test_batch_index]

            # below lines will work if remove lines start after step += 1
            # because of the shape is not match
            accuracy_test = sess.run(accuracy,
                                     feed_dict={x: test_x_batch,
                                                y_: test_y_batch})
            print("test accuracy: %s after %s steps" % (accuracy_test, step))


# todo: difference?
# tf.reshape()   train_x_batch.reshape()

