#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
multiply of two float number
"""
import tensorflow as tf


a = tf.placeholder("float")
b = tf.placeholder("float")

res = tf.multiply(a, b)

with tf.Session() as sess:
    sess.run(res, feed_dict={a: 1.0, b: 2.0})
    print(sess.run(res, feed_dict={a: 1.0, b: 2.0}))

