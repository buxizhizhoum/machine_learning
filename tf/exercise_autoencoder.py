#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
auto encoder
"""
import tensorflow as tf
import numpy as np

from tensorflow.examples.tutorials.mnist import input_data


learning_rate = 0.005
iter_num = 30
batch_size = 128
corruption_level = 0.3

x = tf.placeholder("float", [None, 784])

mask = tf.placeholder("float", [None, 784])

w_max = 4 * np.sqrt(6.0 / 784 * 500)
# weights and biases to encode
# todo: after add maxval and minval in initialization of w
# todo: cost starts to decrease when training, why?
# the impact of initialization?
w = tf.Variable(tf.random_uniform([784, 500],
                                  maxval=w_max,
                                  minval=-w_max))  # means of 500?
b = tf.Variable(tf.zeros([500]))

# weights and biases to decode
w_e = tf.transpose(w)
b_e = tf.Variable(tf.zeros([784]))


def model():
    tmp = mask * x  # what is the affect of this line?
    y_encode_o = tf.nn.sigmoid(tf.add(tf.matmul(tmp, w), b))
    y_decode_o = tf.nn.sigmoid(tf.add(tf.matmul(y_encode_o, w_e), b_e))
    return y_decode_o

y = model()

cost = tf.reduce_sum(tf.square(x - y))
# cost = tf.reduce_sum(tf.pow((x-y), 2))
train_step = tf.train.RMSPropOptimizer(learning_rate).minimize(cost)
# train_step = tf.train.GradientDescentOptimizer(0.02).minimize(cost)

accuracy = tf.reduce_mean(tf.cast(
    tf.equal(tf.argmax(x, 1), tf.argmax(y, 1)), "float"))

init = tf.global_variables_initializer()

if __name__ == "__main__":
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    test_x, test_y = mnist.test.images, mnist.test.labels

    with tf.Session() as sess:
        sess.run(init)

        step = 0
        while step < iter_num:

            for i in range(1000):
                train_x_batch, train_y_batch = mnist.train.next_batch(
                    batch_size)
                mask_np = np.random.binomial(
                    1, 1 - corruption_level, train_x_batch.shape)
                sess.run(train_step,
                         feed_dict={x: train_x_batch, mask: mask_np})

                if i % 100 == 0:
                    accuracy_train = sess.run(accuracy,
                                              feed_dict={x: train_x_batch,
                                                         mask: mask_np})
                    print(accuracy_train)
                    cost_val = sess.run(cost,
                                        feed_dict={x: train_x_batch,
                                                   mask: mask_np})
                    print("cost: %s" % cost_val)

            step += 1
            mask_test_np = np.random.binomial(
                1, 1 - corruption_level, test_x.shape)
            accuracy_test = sess.run(accuracy,
                                     feed_dict={x: test_x, mask: mask_test_np})
            print("accuracy on test data sets: %s after %d of iter"
                  % (accuracy_test, step))






































