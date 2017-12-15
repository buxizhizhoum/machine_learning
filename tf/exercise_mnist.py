#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
exercise of mnist

the model is: tf.softmax(tf.matmul(x, w) + b)

step1:
    build model:
        predict = tf.softmax(tf.matmul(x, w) + b)

        rely on: x = tf.placeholder()
                w = tf.Variable()
                b = tf.Variable()
step2:
    cost function:
        cross_entropy = tf.reduce_mean(y_ * log(predict))

        rely on: y_ = tf.placeholder()

step3:
    training, minimize cost function:
        gradient decent
        tf.train.GradientDescentOptimizer().minimize(cost_function)

step4:
    accuracy:
        predict_accuracy = tf.equal(tf.argmax(predict, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(predict_accuracy))
"""
import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data

learning_rate = 0.003
train_times = 10000

mnist_data_sets = input_data.read_data_sets("MNIST_data/", one_hot=True)

# used when create model in tf.softmax(tf.matmul(x, w) + b)
x = tf.placeholder("float", [None, 784])
# used in cross entropy tf.reduce_sum(y_ * predict)
y_ = tf.placeholder("float", [None, 10])
# the placeholder above should feed in training, with the key x and y_

# todo: choose one way to initialize w and b from below method.
# 1.initialize to zero
# w = tf.Variable(tf.zeros([784, 10]))
# b = tf.Variable(tf.zeros([10]))

# 2.initialize to random normal
# question: why random normal need more iter times to get same accuracy?
# w = tf.Variable(tf.random_normal([784, 10]))
# b = tf.Variable(tf.random_normal([10]))

# 3.initialize to random uniform
# why this move quickly to high accuracy than random normal?
w = tf.Variable(tf.random_uniform([784, 10]))
b = tf.Variable(tf.random_uniform([10]))

# build model
model = tf.matmul(x, w) + b
predict = tf.nn.softmax(model)

# initialize variable, w and b
init = tf.global_variables_initializer()

# create session
sess = tf.Session()
sess.run(init)

# todo: what is the difference between 2 below cross_entropy
# calculate difference between predict and labels
# cross_entropy = tf.reduce_mean(
#     tf.nn.softmax_cross_entropy_with_logits(logits=predict, labels=y_))
# above method seems not as good as below in this case.
# cost with cross entropy
cross_entropy = -tf.reduce_sum(y_ * tf.log(predict))
train_step = tf.train.\
            GradientDescentOptimizer(learning_rate).minimize(cross_entropy)

# calculate accuracy, this is only need when print accuracy
predict_equal = tf.equal(tf.argmax(y_, 1), tf.argmax(predict, 1))
# [correct, incorrect, correct, correct, incorrect]
# = [True, False, True, True, False]
# = [1, 0, 1, 1, 0] # cast transfer True or False to 1 or 0
# with mean of 0.6
accuracy = tf.reduce_mean(tf.cast(predict_equal, "float"))

# train, and print accuracy every 100 time
step = 0
while step < train_times:
    train_x, train_y = mnist_data_sets.train.next_batch(100)
    # provide x, and real label of x
    sess.run(train_step, feed_dict={x: train_x, y_: train_y})
    if step % 100 == 0:
        # print("training...")
        test_x = mnist_data_sets.test.images
        test_y = mnist_data_sets.test.labels
        print(sess.run(accuracy, feed_dict={x: test_x, y_: test_y}))
    step += 1

sess.close()
