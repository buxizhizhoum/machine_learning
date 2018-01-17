#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
single hidden layer nn, learning_rate decay, and moving average of variables.

the accuracy on test data set is 97.96%
"""
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

INPUT_NODE = 784
OUTPUT_NODE = 10
LAYER_1_NODE = 784

BATCH_SIZE = 100

LEARNING_RATE_BASE = 0.1
LEARNING_RATE_DECAY = 0.99

REGULARIZATION_RATE = 0.001
TRAINING_STEPS = 30000
MOVING_AVERAGE_DECAY = 0.99


def inference(input_tensor, avg_class, weights_1, biases_1, weights_2,
              biases_2):
    if avg_class is None:
        layer_1 = tf.nn.relu(tf.matmul(input_tensor, weights_1) + biases_1)
        # softmax will be add when calculate loss function,
        # so there is no softmax
        layer_2 = tf.matmul(layer_1, weights_2) + biases_2
    else:
        layer_1 = tf.nn.relu(
            tf.matmul(input_tensor, avg_class.average(weights_1))
            + avg_class.average(biases_1)
        )
        layer_2 = tf.matmul(layer_1, avg_class.average(weights_2)) \
                  + avg_class.average(biases_2)
    return layer_2


def train(mnist):
    x = tf.placeholder(tf.float32, [None, INPUT_NODE], name="input")
    y_ = tf.placeholder(tf.float32, [None, OUTPUT_NODE], name="output")

    weights_1 = tf.Variable(
        tf.truncated_normal([INPUT_NODE, LAYER_1_NODE], stddev=0.1))
    biases_1 = tf.Variable(tf.constant(0.1, shape=[LAYER_1_NODE]))

    weights_2 = tf.Variable(
        tf.truncated_normal([LAYER_1_NODE, OUTPUT_NODE], stddev=0.1))
    biases_2 = tf.Variable(tf.constant(0.1, shape=[OUTPUT_NODE]))

    # calculate forward propagation result
    y = inference(x, None, weights_1, biases_1, weights_2, biases_2)

    global_step = tf.Variable(0, trainable=False)
    # when num_steps is given, the update at the beginning is faster
    variable_averages = tf.train.ExponentialMovingAverage(
        MOVING_AVERAGE_DECAY, global_step)
    variable_averages_op = variable_averages.apply(tf.trainable_variables())

    # pass variable_averages to inference when necessary
    y_average = inference(x, variable_averages, weights_1, biases_1,
                          weights_2, biases_2)
    # sparse_softmax_cross_entropy_with_logits is faster than
    # softmax_corss_entropy_with_logits
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=y, labels=tf.argmax(y_, 1))
    # cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
    #     logits=y, labels=y_)
    cross_entropy_mean = tf.reduce_mean(cross_entropy)

    regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)
    # regular is related to weights only
    regularization = regularizer(weights_1) + regularizer(weights_2)

    loss = cross_entropy_mean + regularization

    decay_steps = mnist.train.num_examples / BATCH_SIZE
    learning_rate = tf.train.exponential_decay(
        learning_rate=LEARNING_RATE_BASE,
        global_step=global_step,
        decay_steps=decay_steps,
        decay_rate=LEARNING_RATE_DECAY)
    # 优化使用正则化的损失函数loss
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(
        loss, global_step=global_step)
    # 下面是在优化时没有使用正则化的损失函数，会导致过拟合
    # train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(
    #     cross_entropy_mean, global_step=global_step)
    # 一次完成更新参数和更新参数的滑动平均值两个操作
    train_op = tf.group(train_step, variable_averages_op)
    # 与上一行代码等价
    # with tf.control_dependencies([train_step, variable_averages_op]):
    #     train_op = tf.no_op(name="train")

    correct_prediction = tf.equal(tf.argmax(y_average, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        validate_feed = {x: mnist.validation.images,
                         y_: mnist.validation.labels}

        test_feed = {x: mnist.test.images,
                     y_: mnist.test.labels}

        for i in range(TRAINING_STEPS):
            xs, ys = mnist.train.next_batch(BATCH_SIZE)
            sess.run(train_op, feed_dict={x: xs, y_: ys})

            if i % 100 == 0:
                validate_acc = sess.run(accuracy, feed_dict=validate_feed)
                print("%s: validate accuracy: %s" % (i, validate_acc))
                test_acc = sess.run(accuracy, feed_dict=test_feed)
                print("%s: test accuracy: %s" % (i, test_acc))
                # accuracy on training batch
                print(sess.run(accuracy, feed_dict={x: xs, y_: ys}))

        test_acc = sess.run(accuracy, feed_dict=test_feed)
        print("test accuracy: %s" % test_acc)


def main(argv=None):
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    train(mnist)


if __name__ == "__main__":
    tf.app.run()
    # main()












