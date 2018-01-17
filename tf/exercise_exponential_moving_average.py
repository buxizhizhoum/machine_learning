#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
when apply average, the new value is a value between initial value and
updated value
"""
import tensorflow as tf
# v1 is the variable to apply exponential moving average
v1 = tf.Variable(0.0, tf.float32)
# assume step is the steps to train in nn
step = tf.Variable(0, trainable=False)
# a class of moving average
ema = tf.train.ExponentialMovingAverage(0.99, step)
# an operation to do moving average
maintain_average_op = ema.apply([v1])

init = tf.initialize_all_variables()

with tf.Session() as sess:
    sess.run(init)

    print("when v1 is 0:")
    print(sess.run([v1, ema.average(v1)]))

    # update value of v1
    sess.run(tf.assign(v1, 5.0))
    sess.run(maintain_average_op)
    print(sess.run([v1, ema.average(v1)]))

    # update the value of step
    sess.run(tf.assign(step, 1000))
    sess.run(tf.assign(v1, 10))

    sess.run(maintain_average_op)
    print(sess.run([v1, ema.average(v1)]))
    # run again
    sess.run(maintain_average_op)
    print(sess.run([v1, ema.average(v1)]))



