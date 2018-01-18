#!/usr/bin/python
# -*- coding: utf-8 -*-
import os
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


import mnist_inference


BATCH_SIZE = 100

LEARNING_RATE_BASE = 0.1
LEARNING_RATE_DECAY = 0.99

REGULARIZATION_RATE = 0.001
TRAINING_STEPS = 30000
MOVING_AVERAGE_DECAY = 0.99

MODEL_SAVE_PATH = "/home/buxizhizhoum/tf_model/"
MODEL_NAME = "model_02.ckpt"


def train(mnist):
    x = tf.placeholder(
        tf.float32,
        [None, mnist_inference.INPUT_NODE],
        name="x_input")
    y_ = tf.placeholder(
        tf.float32,
        [None, mnist_inference.OUTPUT_NODE],
        name="y_label")

    regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)
    y = mnist_inference.inference(x, regularizer)

    global_step = tf.Variable(0, trainable=False)
    variable_averages = tf.train.ExponentialMovingAverage(
        MOVING_AVERAGE_DECAY, global_step)
    variable_average_op = variable_averages.apply(tf.trainable_variables())

    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=y, labels=tf.arg_max(y_, 1))
    cross_entropy_mean = tf.reduce_mean(cross_entropy)

    loss = cross_entropy_mean + tf.add_n(tf.get_collection("losses"))
    learning_rate = tf.train.exponential_decay(
        learning_rate=LEARNING_RATE_BASE,
        global_step=global_step,
        decay_steps=mnist.train.num_examples / BATCH_SIZE,
        decay_rate=LEARNING_RATE_DECAY)

    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(
        loss, global_step=global_step)

    train_op = tf.group(train_step, variable_average_op)
    init = tf.global_variables_initializer()

    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(init)

        for i in range(TRAINING_STEPS):
            xs, ys = mnist.train.next_batch(BATCH_SIZE)
            _, loss_value, step = sess.run(
                [train_op, loss, global_step], feed_dict={x: xs, y_: ys})

            if i % 100 == 0:
                print("%s loss: %s" % (i, loss_value))
                filename = os.path.join(MODEL_SAVE_PATH, MODEL_NAME)
                saver.save(sess, filename, global_step=global_step)


def main(argv=None):
    mnist = input_data.read_data_sets("../MNIST_data/", one_hot=True)
    train(mnist)


if __name__ == "__main__":
    tf.app.run()
















