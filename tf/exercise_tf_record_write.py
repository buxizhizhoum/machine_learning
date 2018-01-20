#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data


def _int64_feature(value):
    """
    生成整数型的属性
    :param value:
    :return:
    """
    int64_list = tf.train.Int64List(value=[value])
    res = tf.train.Feature(int64_list=int64_list)
    return res


def _bytes_feature(value):
    """
    生成字符串型的属性
    :param value:
    :return:
    """
    bytes_list = tf.train.BytesList(value=[value])
    res = tf.train.Feature(bytes_list=bytes_list)
    return res


if __name__ == "__main__":
    mnist = input_data.read_data_sets(
        "MNIST_data/", dtype=tf.uint8, one_hot=True)

    images = mnist.train.images
    # both labels and pixels could be regarded as an feature
    labels = mnist.train.labels
    pixels = images.shape[1]

    num_examples = mnist.train.num_examples

    filename = "/home/buxizhizhoum/tf_records/record_01"
    writer = tf.python_io.TFRecordWriter(filename)

    for index in range(num_examples):
        # 图像转换为字符串
        image_row = images[index].tostring()
        # 像素，label，图像的字符串三者生成一个feature字典
        feature_dict = {"pixels": _int64_feature(pixels),
                        "label": _int64_feature(np.argmax(labels[index])),
                        "image_row": _bytes_feature(image_row)}
        example = tf.train.Example(
            features=tf.train.Features(feature=feature_dict))

        writer.write(example.SerializeToString())
    writer.close()
