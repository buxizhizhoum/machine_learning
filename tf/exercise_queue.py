#!/usr/bin/python
# -*- coding: utf-8 -*-

import tensorflow as tf

q = tf.FIFOQueue(10, tf.int32)

init = q.enqueue(1)
enqueue_many = q.enqueue_many([[2, 3, 4, 5, 6, 7]])

x = q.dequeue()

y = x + 1

with tf.Session() as sess:
    sess.run(init)
    sess.run(enqueue_many)

    for i in range(7):
        print(sess.run(y))






