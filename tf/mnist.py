#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
rewrite official tutorial of mnist into a class to make it more compact.
copyright belong to its original author.
"""
import tensorflow as tf


# todo initialize weight with random number
class MNIST(object):
    """
    rewrite official tutorial of mnist, to make it more compact.
    """
    def __init__(self, X):
        """
        X is the training set,

        X.train.images.shape[1] is to extract the col number of train input
        set, that is to say, if X.train.image is a matrix with dimension m x n,
        X.train.images.shape[1] extract the number n.

        X.train.labels.shape[1] is to extract the col number of label set.
        :param X:
        """
        self.X = X
        self.x = tf.placeholder("float", [None, X.train.images.shape[1]])
        # W and b is the weight and bias
        self.W = tf.Variable(tf.zeros([X.train.images.shape[1],
                                       X.train.labels.shape[1]]))
        self.b = tf.Variable(tf.zeros([X.train.labels.shape[1]]))
        # create the model
        self.y = tf.nn.softmax(tf.matmul(self.x, self.W) + self.b)
        self.y_ = tf.placeholder("float", [None, X.train.labels.shape[1]])
        self.sess = tf.Session()

    def train_step(self):
        cross_entropy = -tf.reduce_sum(self.y_ * tf.log(self.y))
        train_step = tf.train.GradientDescentOptimizer(0.01).minimize(
            cross_entropy)
        return train_step

    def initialize(self):
        """
        initialize, at the beginning, create of sess is put here,
        it leads very low accuracy, which is less than 0.01,
        then sess initialization is moved to __init__()
        :return:
        """
        init = tf.initialize_all_variables()
        # sess = tf.Session()
        self.sess.run(init)
        # return sess

    def train(self, iter_num=1000):
        train_step = self.train_step()
        for i in range(iter_num):
            batch_xs, batch_ys = self.X.train.next_batch(100)
            self.sess.run(train_step, feed_dict={self.x: batch_xs,
                                                 self.y_: batch_ys})

    def accuracy(self):
        correct_prediction = tf.equal(tf.arg_max(self.y, 1),
                                      tf.arg_max(self.y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        return self.sess.run(accuracy,
                             feed_dict={self.x: self.X.test.images,
                                        self.y_: self.X.test.labels})


def main():
    mnist_obj = MNIST(mnist)

    mnist_obj.initialize()
    mnist_obj.train()
    print(mnist_obj.accuracy())


if __name__ == "__main__":
    import input_data
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

    main()

    # below is the code from tensorflow official site
    # x = tf.placeholder("float", [None, 784])
    # W = tf.Variable(tf.zeros([784, 10]))
    # b = tf.Variable(tf.zeros([10]))
    #
    # # predict y
    # y = tf.nn.softmax(tf.matmul(x, W) + b)
    #
    # # real y
    # y_ = tf.placeholder("float", [None, 10])
    #
    # cross_entropy = -tf.reduce_sum(y_ * tf.log(y))
    # train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)
    #
    # init = tf.initialize_all_variables()
    # sess = tf.Session()
    # sess.run(init)
    #
    # for i in range(1000):
    #     # x_batch, y_batch = mnist.train.next_batch(100)
    #     # sess.run(train_step, feed_dict={x: x_batch, y: y_batch})
    #     batch_xs, batch_ys = mnist.train.next_batch(100)
    #     sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
    #
    #
    # correct_prediction = tf.equal(tf.arg_max(y, 1), tf.arg_max(y_, 1))
    # accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    # print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
