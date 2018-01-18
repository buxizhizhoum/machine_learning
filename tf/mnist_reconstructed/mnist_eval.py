#!/usr/bin/python
# -*- coding: utf-8 -*-

import tensorflow as tf
import time

from tensorflow.examples.tutorials.mnist import input_data


import mnist_inference
import mnist_train


EVAL_INTERVAL_SEC = 10


def evaluate(mnist):
    with tf.Graph().as_default() as g:
        x = tf.placeholder(
            tf.float32,
            [None, mnist_inference.INPUT_NODE],
            name="x_input")
        y_ = tf.placeholder(
            tf.float32,
            [None, mnist_inference.OUTPUT_NODE],
            name="y_label")

        validate_feed = {x: mnist.validation.images,
                         y_: mnist.validation.labels}
        # no regular is need during test
        y = mnist_inference.inference(x, None)

        correct_prediction = tf.equal(tf.arg_max(y, 1), tf.arg_max(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction,  tf.float32))

        variables_average = tf.train.ExponentialMovingAverage(
            mnist_train.MOVING_AVERAGE_DECAY)
        variables_to_restore = variables_average.variables_to_restore()
        saver = tf.train.Saver(variables_to_restore)

        while True:
            with tf.Session() as sess:
                # get the newest model name automatically
                ckpt = tf.train.get_checkpoint_state(
                    mnist_train.MODEL_SAVE_PATH)
                if ckpt and ckpt.model_checkpoint_path:
                    saver.restore(sess, ckpt.model_checkpoint_path)
                    global_step = ckpt.model_checkpoint_path \
                        .split("/")[-1].split("-")[-1]
                    accuracy_score = sess.run(
                        accuracy, feed_dict=validate_feed)
                    print("%s: accuracy: %s" % (global_step, accuracy_score))
                else:
                    print("no ckpt file found!")
                    return

            time.sleep(EVAL_INTERVAL_SEC)


def main(argv=None):
    mnist = input_data.read_data_sets("../MNIST_data/", one_hot=True)
    evaluate(mnist)


if __name__ == "__main__":
    tf.app.run()





