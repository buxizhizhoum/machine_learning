#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
neural network, single hidden layer.

the accuracy of single hidden layer nn is more than 93% better than softmax
example on official website.

"""
import tensorflow as tf
import numpy as np

from tensorflow.examples.tutorials.mnist import input_data


learning_rate = 0.05
iter_num = 30
batch_size = 128

x = tf.placeholder("float", [None, 784])
y_ = tf.placeholder("float", [None, 10])
# weights of input layer
w_in = tf.Variable(tf.random_normal([784, 784], stddev=0.01))
# weights of hiddern layer
w_hidden = tf.Variable(tf.random_normal([784, 10], stddev=0.01))
# biases of input layer
b_in = tf.Variable(tf.random_uniform([784]))
# biases of hidden layer
b_hidden = tf.Variable(tf.random_uniform([10]))


def model():
    # build model
    hidden_in_tmp = tf.add(tf.matmul(x, w_in), b_in)
    # hidden_in = tf.nn.sigmoid(hidden_in_tmp)  # not work fine.
    hidden_in = tf.nn.softmax(hidden_in_tmp)

    hidden_out_tmp = tf.add(tf.matmul(hidden_in, w_hidden), b_hidden)
    hidden_out = tf.nn.softmax(hidden_out_tmp)

    return hidden_out

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
            for i in range(1000):
                train_x_batch, train_y_batch = mnist.train.next_batch(128)
                sess.run(train_step,
                         feed_dict={x: train_x_batch, y_: train_y_batch})
                # run every 100 times
                if i % 100 == 0:
                    # accuracy on train data set
                    accuracy_train = sess.run(accuracy,
                                              feed_dict={x: train_x_batch,
                                                         y_: train_y_batch})
                    print(accuracy_train)

            step += 1
            # calculate accuracy on test data set
            accuracy_test = sess.run(accuracy,
                                     feed_dict={x: test_x, y_: test_y})
            print("test accuracy: %s", accuracy_test)
