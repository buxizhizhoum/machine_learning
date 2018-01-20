#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
not passed.
"""
import tensorflow as tf


filename = "/home/buxizhizhoum/tf_records/record_01"
reader = tf.TFRecordReader()
# writer = tf.python_io.TFRecordWriter(filename)

filename_queue = tf.train.string_input_producer([filename])

_, serialized_example = reader.read(filename_queue)

feature_dict = {"image_row": tf.FixedLenFeature([], tf.string),
                "pixels": tf.FixedLenFeature([], tf.int64),
                "label": tf.FixedLenFeature([], tf.int64)}
features = tf.parse_single_example(serialized_example, features=feature_dict)

images = tf.decode_raw(features["image_row"], tf.uint8)
labels = tf.cast(features["label"], tf.int32)
pixels = tf.cast(features["pixels"], tf.int32)


sess = tf.Session()

coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess=sess, coord=coord)

for i in range(10):
    image, label, pixel = sess.run([images, labels, pixels])
    print(image)
