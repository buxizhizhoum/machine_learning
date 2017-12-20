#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
neural network with 2 hidden layer
"""
import tensorflow as tf
import numpy as np

from tensorflow.examples.tutorials.mnist import input_data


learning_rate = 0.001
iter_num = 50
batch_size = 128  # how to choose?

x = tf.placeholder("float", [None, 784])  # input
y_ = tf.placeholder("float", [None, 10])  # label

# weights of input layer
w_i = tf.Variable(tf.random_normal([784, 784], stddev=0.01))
# weights of first hidden layer
w_h_1 = tf.Variable(tf.random_normal([784, 784], stddev=0.01))
# weights of second hidden layer
w_h_2 = tf.Variable(tf.random_normal([784, 10], stddev=0.01))
# biases of input layer
b_i = tf.Variable(tf.random_uniform([784]))
# biases of first hidden layer
b_h_1 = tf.Variable(tf.random_uniform([784]))
# biases of second hidden layer
b_h_2 = tf.Variable(tf.random_uniform([10]))


def model():
    # output of input layer
    y_i_tmp = tf.add(tf.matmul(x, w_i), b_i)
    y_i = tf.nn.softmax(y_i_tmp)  # or other active function

    # output of first hidden layer
    y_w_1_tmp = tf.add(tf.matmul(y_i, w_h_1), b_h_1)
    y_w_1 = tf.nn.softmax(y_w_1_tmp)

    # output of second hidden layer
    y_w_2_tmp = tf.add(tf.matmul(y_w_1, w_h_2), b_h_2)
    y_w_2 = tf.nn.softmax(y_w_2_tmp)

    return y_w_2


y = model()

# cost function, cross entropy
cost = -tf.reduce_sum(y_ * tf.log(y))
# train_step = tf.train.AdamOptimizer(learning_rate).minimize(cost)
# train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
# two train_steps above not work well, why?
train_step = tf.train.RMSPropOptimizer(learning_rate).minimize(cost)

# calculate accuracy
predict_eql = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(predict_eql, "float"))

# initialize all variables
init = tf.global_variables_initializer()

if __name__ == "__main__":
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    # test data set
    test_x, test_y = mnist.test.images, mnist.test.labels

    step = 0

    with tf.Session() as sess:
        sess.run(init)
        while step < iter_num:
            for i in range(1000):
                # train batch by batch
                train_x_batch, train_y_batch \
                    = mnist.train.next_batch(batch_size)
                sess.run(train_step,
                         feed_dict={x: train_x_batch,
                                    y_: train_y_batch})

                if i % 100 == 0:
                    accuracy_train = sess.run(accuracy,
                                              feed_dict={x: train_x_batch,
                                                         y_: train_y_batch})
                    print(accuracy_train)

            step += 1
            accuracy_test = sess.run(accuracy,
                                     feed_dict={x: test_x,
                                                y_: test_y})
            print("accuracy on test data sets: %s after %d of iter"
                  % (accuracy_test, step))





