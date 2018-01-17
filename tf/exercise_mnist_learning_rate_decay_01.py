#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
single hidden layer nn, learning_rate decay, variable scope
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


def inference(input_tensor, reuse=False):
    """
    build model, each variable is under the scope of its layer.
    :param input_tensor:
    :param reuse:
    :return:
    """
    with tf.variable_scope("layer_1", reuse=reuse):
        weights = tf.get_variable(
            "weight",
            shape=[INPUT_NODE, LAYER_1_NODE],
            initializer=tf.truncated_normal_initializer(stddev=0.1))
        biases = tf.get_variable(
            "biases",
            shape=[LAYER_1_NODE],
            initializer=tf.constant_initializer(0.01))
        layer_1 = tf.nn.relu(tf.matmul(input_tensor, weights) + biases)

    with tf.variable_scope("layer_2", reuse=reuse):
        weights = tf.get_variable(
            "weight",
            shape=[LAYER_1_NODE, OUTPUT_NODE],
            initializer=tf.truncated_normal_initializer(stddev=0.1))
        biases = tf.get_variable(
            "biases",
            shape=[OUTPUT_NODE],
            initializer=tf.constant_initializer(0.01))
        layer_2 = tf.matmul(layer_1, weights) + biases

    return layer_2


def train(mnist):
    x = tf.placeholder(tf.float32, [None, INPUT_NODE], name="input")
    y_ = tf.placeholder(tf.float32, [None, OUTPUT_NODE], name="output")

    # calculate forward propagation result
    y = inference(x)

    global_step = tf.Variable(0, trainable=False)
    # when num_steps is given, the update at the beginning is faster

    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=y, labels=tf.argmax(y_, 1))
    cross_entropy_mean = tf.reduce_mean(cross_entropy)

    decay_steps = mnist.train.num_examples / BATCH_SIZE
    learning_rate = tf.train.exponential_decay(
        learning_rate=LEARNING_RATE_BASE,
        global_step=global_step,
        decay_steps=decay_steps,
        decay_rate=LEARNING_RATE_DECAY)
    # 下面是在优化时没有使用正则化的损失函数，会导致过拟合
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(
        cross_entropy_mean, global_step=global_step)
    train_op = tf.group(train_step)
    # 与上一行代码等价
    # with tf.control_dependencies([train_step, variable_averages_op]):
    #     train_op = tf.no_op(name="train")

    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
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
