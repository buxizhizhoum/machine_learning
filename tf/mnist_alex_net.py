#!/usr/bin/python
# -*- coding: utf-8 -*-

import tensorflow as tf


def conv_2d(name, x, W, b, strides=1):
    x = tf.nn.conv2d(x, W,
                     strides=[1, strides, strides, 1],
                     padding="SAME")
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x, name=name)


def max_pool_2d(name, x, k=2):
    return tf.nn.max_pool(x,
                          ksize=[1, k, k, 1],
                          strides=[1, k, k, 1],
                          padding="SAME",
                          name=name)


def norm(name, l_input, l_size=4):
    return tf.nn.lrn(l_input, l_size,
                     bias=0.1, alpha=0.001 / 9.0,
                     beta=0.75, name=name)


def alex_net(x, weights, biases, drop_out):
    # Reshape
    x = tf.reshape(x, shape=[-1, 28, 28, 1])

    # 1st layer
    conv_1 = conv_2d("conv_1", x, weights["wc1"], biases["bc1"])
    pool_1 = max_pool_2d("pool_1", conv_1, k=2)
    norm_1 = norm("norm_1", pool_1, l_size=4)

    # 2nd layer
    conv_2 = conv_2d("conv_2", conv_1, weights["wc2"], biases["bc2"])
    pool_2 = max_pool_2d("pool_2", conv_2, k=2)
    norm_2 = norm("norm_2", pool_2, l_size=4)

    # 3rd layer
    conv_3 = conv_2d("conv_3", norm_2, weights["wc3"], biases["bc3"])
    pool_3 = max_pool_2d("pool_3", conv_3, k=2)
    norm_3 = norm("norm_3", pool_3, l_size=4)

    # 4th layer
    conv_4 = conv_2d("conv_4", norm_3, weights["wc4"], biases["bc4"])
    pool_4 = max_pool_2d("pool_4", conv_4, k=2)
    norm_4 = norm("norm_4", pool_4, l_size=4)

    # 5th layer
    conv_5 = conv_2d("conv_5", norm_3, weights["wc5"], biases["bc5"])
    pool_5 = max_pool_2d("pool_5", conv_5, k=2)
    norm_5 = norm("norm_5", pool_5, l_size=4)

    # full connect layer
    fc1 = tf.reshape(norm_5, [-1, weights["wd1"].get_shape().as_list()[0]])
    fc1 = tf.add(tf.matmul(fc1, weights["wd1"]), biases["bd1"])
    fc1 = tf.nn.relu(fc1)

    # dropout
    fc1 = tf.nn.dropout(fc1, drop_out)

    # full connect layer 2
    fc2 = tf.reshape(fc1, [-1, weights["wd1"].get_shape().as_list()[0]])
    fc2 = tf.add(tf.matmul(fc2, weights["wd1"]), biases["bd1"])
    fc2 = tf.nn.relu(fc2)

    # dropout
    fc2 = tf.nn.dropout(fc2, drop_out)

    # output layer
    out = tf.add(tf.matmul(fc2, weights["out"]), biases["out"])
    return out


if __name__ == "__main__":
    import input_data
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

    learning_rate = 0.001
    training_iter = 20000
    batch_size = 128
    display_step = 10

    n_input = 784
    n_classes = 10
    dropout = 0.75

    x = tf.placeholder("float", [None, n_input])  # tf.float32?
    y = tf.placeholder("float", [None, n_classes])
    keep_prob = tf.placeholder("float")

    weights = {
        'wc1': tf.Variable(tf.random_normal([11, 11, 1, 96])),
        'wc2': tf.Variable(tf.random_normal([5, 5, 96, 256])),
        'wc3': tf.Variable(tf.random_normal([3, 3, 256, 384])),
        'wc4': tf.Variable(tf.random_normal([3, 3, 384, 384])),
        'wc5': tf.Variable(tf.random_normal([3, 3, 384, 256])),
        'wd1': tf.Variable(tf.random_normal([4 * 4 * 256, 4096])),
        'wd2': tf.Variable(tf.random_normal([4096, 4096])),
        'out': tf.Variable(tf.random_normal([4096, 10]))
    }
    biases = {
        'bc1': tf.Variable(tf.random_normal([96])),
        'bc2': tf.Variable(tf.random_normal([256])),
        'bc3': tf.Variable(tf.random_normal([384])),
        'bc4': tf.Variable(tf.random_normal([384])),
        'bc5': tf.Variable(tf.random_normal([256])),
        'bd1': tf.Variable(tf.random_normal([4096])),
        'bd2': tf.Variable(tf.random_normal([4096])),
        'out': tf.Variable(tf.random_normal([n_classes]))
    }

    # build model
    predict = alex_net(x, weights, biases, keep_prob)
    # loss function and optimizer
    cost = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(logits=predict, labels=y))
    optimizer = tf.train.AdamOptimizer(
        learning_rate=learning_rate).minimize(cost)

    correct_prediction = tf.equal(tf.argmax(predict, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # below is to train and evaluate model
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        step = 0
        while step * batch_size < training_iter:
            batch_x, batch_y = mnist.train.next_batch(batch_size)
            sess.run(optimizer,
                     feed_dict={x: batch_x, y: batch_y, keep_prob: dropout})
            print(step)
            if step % display_step == 0:
                loss, accuracy_training = sess.run([cost, accuracy],
                                                   feed_dict={x: batch_x,
                                                   y: batch_y,
                                                   keep_prob: 1.0})

                print("iter %s, " "Minibatch loss = %.6f, "
                      "Training accurancy = %.6f"
                      % (str(step*batch_size), loss, accuracy_training))
            step += 1
        print("Training finished!")

        test_accuracy = sess.run(accuracy,
                                 feed_dict={x: mnist.test.images[:256],
                                            y: mnist.test.labels[:256],
                                            keep_prob: dropout})

        print("Test accuracy: %s" % test_accuracy)
