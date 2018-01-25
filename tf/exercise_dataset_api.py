#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
examples of tensorflow Dataset api
ref: https://zhuanlan.zhihu.com/p/30751039
"""
import numpy as np
import tensorflow as tf


NUMBER = 10  # number of data


data = np.array(np.arange(NUMBER))
# 创建了一个dataset
# 其实，tf.data.Dataset.from_tensor_slices的功能不止如此，它的真正作用是
# 切分传入Tensor的第一个维度，生成相应的dataset。
dataset = tf.data.Dataset.from_tensor_slices(data)
# 语句iterator = dataset.make_one_shot_iterator()从dataset中实例化了一个
# Iterator，这个Iterator是一个“one shot iterator”，即只能从头到尾读取一次
iterator = dataset.make_one_shot_iterator()
# one_element = iterator.get_next()表示从iterator里取出一个元素,
# one_element只是一个Tensor，并不是一个实际的值。
# 调用sess.run(one_element)后，才能真正地取出一个值
element = iterator.get_next()

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    for i in range(NUMBER):
        data_element = sess.run(element)
        print(data_element)
    # 因为iterator数据已经取完了，下面这一句会抛出OutOfRangeError
    # sess.run(element)


"""
another example
"""
# 其实，tf.data.Dataset.from_tensor_slices的功能不止如此，它的真正作用是切分
# 传入Tensor的第一个维度，生成相应的dataset。
data_1 = np.random.uniform(size=(5, 9))
# 切分它形状上的第一个维度，最后生成的dataset中一个含有5个元素，
# 每个元素的形状是(9, )，即每个元素是矩阵的一行。
dataset_1 = tf.data.Dataset.from_tensor_slices(data_1)
iterator_1 = dataset_1.make_one_shot_iterator()
element_1 = iterator_1.get_next()

init_1 = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init_1)
    for _ in range(5):
        data_element_1 = sess.run(element_1)
        print(data_element_1)


"""
example_3
"""
data_2 = {"a": np.arange(5), "b": np.random.uniform(size=(5, 9))}
dataset_2 = tf.data.Dataset.from_tensor_slices(data_2)
iterator_2 = dataset_2.make_one_shot_iterator()
element_2 = iterator_2.get_next()

init_2 = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init_2)
    for _ in range(5):
        data_element_2 = sess.run(element_2)
        print(data_element_2)


"""
exapmle 4
"""
data_3 = {"a": np.arange(5), "b": np.random.uniform(size=(5, 9))}
dataset_3 = tf.data.Dataset.from_tensor_slices(data_2)
# batch就是将多个元素组合成batch
dataset_3 = dataset_3.batch(1)
iterator_3 = dataset_3.make_one_shot_iterator()
element_3 = iterator_3.get_next()

init_3 = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init_3)
    for _ in range(5):
        data_element_3 = sess.run(element_3)
        print("element 3: %s" % data_element_3)

"""
example 5

initializable iterator必须要在使用前通过sess.run()来初始化。
使用initializable iterator，可以将placeholder代入Iterator中，这可以方便我们通过
参数快速定义新的Iterator。一个简单的initializable iterator使用示例
"""
limit = tf.placeholder(dtype=tf.int32, shape=[])

dataset = tf.data.Dataset.from_tensor_slices(tf.range(start=0, limit=limit))

iterator = dataset.make_initializable_iterator()
next_element = iterator.get_next()

with tf.Session() as sess:
    # 区别在这里，可以feed 一个数到limit
    sess.run(iterator.initializer, feed_dict={limit: 10})
    for i in range(10):
      value = sess.run(next_element)
      assert i == value


"""
initializable iterator还有一个功能：读入较大的数组。

在使用tf.data.Dataset.from_tensor_slices(array)时，实际上发生的事情是将array
作为一个tf.constants保存到了计算图中。当array很大时，会导致计算图变得很大，
给传输、保存带来不便。这时，我们可以用一个placeholder取代这里的array，并使用
initializable iterator，只在需要时将array传进去，这样就可以避免把大数组保存在
图里，示例代码为（来自官方例程）
"""
# 从硬盘中读入两个Numpy数组
# with np.load("/var/data/training_data.npy") as data:
#     features = data["features"]
#     labels = data["labels"]
#
# features_placeholder = tf.placeholder(features.dtype, features.shape)
# labels_placeholder = tf.placeholder(labels.dtype, labels.shape)
#
# dataset = tf.data.Dataset.from_tensor_slices(
#     (features_placeholder, labels_placeholder))
# iterator = dataset.make_initializable_iterator()
# sess.run(iterator.initializer, feed_dict={features_placeholder: features,
#                                           labels_placeholder: labels})

