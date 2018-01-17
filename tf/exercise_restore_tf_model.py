#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
save model of tensorflow

the mainly used is the tf.train.Saver() class
"""
import tensorflow as tf


var_1 = tf.Variable(tf.constant(1.0, shape=[1]), name="var_1")
var_2 = tf.Variable(tf.constant(1.0, shape=[1]), name="var_2")

# init = tf.global_variables_initializer()
res = tf.add(var_1, var_2)

# this is the core class used when saving model
saver = tf.train.Saver()

with tf.Session() as sess:
    # Attention: there is no initialization
    # sess.run(init)
    saver.restore(sess, "/home/buxizhizhoum/tf_model/model_01.ckpt")
    print(sess.run(res))

