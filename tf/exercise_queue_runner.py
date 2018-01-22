#!/usr/bin/python
# -*- coding: utf-8 -*-


import tensorflow as tf


queue = tf.FIFOQueue(100, tf.float32)

enqueue_op = queue.enqueue(tf.random_normal([1]))
queue_runner = tf.train.QueueRunner(queue, [enqueue_op] * 5)
tf.train.add_queue_runner(queue_runner)

out_op = queue.dequeue()

with tf.Session() as sess:
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    for _ in range(3):
        print(sess.run(out_op)[0])

    coord.request_stop()
    coord.join()
