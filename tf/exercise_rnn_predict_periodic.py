#!/usr/bin/python
# -*- coding: utf-8 -*-

import datetime
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


HIDDEN_SIZE = 30
NUM_LAYERS = 2
TIME_STEPS = 10  # 训练序列长度
TRAINING_STEPS = 10000
BATCH_SIZE = 32
TRAINING_EXAMPLES = 10000
TEST_EXAMPLES = 1000  # 测试数据个数
SAMPLE_GAP = 0.01

TRAINING_SLICE = 1000

FILE_NAME = "/home/buxizhizhoum/1-Work/Documents/2-Codes/learning/tf/training_data/training_data.csv"


def read_data(filename):
    data = pd.read_csv(filename)
    res = data.loc[:, ["stat_time", "pi"]]
    # print(res)
    return res


def get_a_cell(lstm_size, keep_prob):
    lstm = tf.nn.rnn_cell.BasicLSTMCell(lstm_size)
    drop = tf.nn.rnn_cell.DropoutWrapper(lstm, output_keep_prob=keep_prob)
    return drop
    # return lstm


def lstm_model(x, y_, is_training):
    rnn_layers = [get_a_cell(HIDDEN_SIZE, 1) for _ in range(NUM_LAYERS)]
    multi_rnn_cell = tf.nn.rnn_cell.MultiRNNCell(rnn_layers)
    # dimension of output is: [batch_size, max_time, cell.output_size]
    outputs, final_state = tf.nn.dynamic_rnn(
        multi_rnn_cell, x, dtype=tf.float32)
    output = outputs[:, -1, :]

    # prediction, loss = learn.models.linear_regression(output, y)
    prediction = tf.contrib.layers.fully_connected(
        output, 1, activation_fn=None)

    if not is_training:
        return prediction, None, None

    loss = tf.losses.mean_squared_error(labels=y_, predictions=prediction)
    train_op = tf.contrib.layers.optimize_loss(
        loss,
        tf.train.get_global_step(),
        optimizer="Adagrad",
        learning_rate=0.1)
    return prediction, loss, train_op


def datetime_timestamp(dt):
    """
    convert datetime to timestamp
    :param dt: datetime
    :return: timestamp
    """
    timestamp = dt.replace().timestamp()
    return timestamp


def generate_data(sequence, start=0):
    """
    produce data used to train model.
    :param sequence:
    :return:
    """
    x = []
    y = []
    start = start
    end = start + (len(sequence) - TIME_STEPS)
    for i in range(start, end):
        x.append([sequence[i: i + TIME_STEPS]])
        y.append([sequence[i]])
    return np.array(x, dtype=np.float32), np.array(y, dtype=np.float32)


if __name__ == "__main__":
    raw_data = read_data(FILE_NAME)
    seq_x = raw_data["stat_time"]
    # print(train_x)
    seq_x = [datetime.datetime.strptime(item, "%Y-%m-%d %H:%M:%S") for item in seq_x]
    seq_x = [datetime_timestamp(item) for item in seq_x]
    seq_y = raw_data["pi"].tolist()  # change series to list
    # slice to get training and test data set
    train_seq_x, train_seq_y = seq_x[:TRAINING_SLICE], seq_y[:TRAINING_SLICE]
    test_seq_x, test_seq_y = seq_x[TRAINING_SLICE:], seq_y[TRAINING_SLICE:]
    print(len(test_seq_y))
    print(test_seq_y[1: 10])

    plt.figure("raw_data")
    # plt.plot(seq_x, seq_y)
    plt.plot(seq_y, label="raw_data")
    plt.legend()
    plt.show()
    # produce sequence
    # train_x and train_y is not a point pair like (x, y) from two sequence,
    # it is points pair line (y1, y2, y3...yn, y_) from one sequence
    # which is time serialized.
    # train_x, train_y = generate_data(train_seq_y)
    train_x, train_y = generate_data(seq_y)
    data_set = tf.data.Dataset.from_tensor_slices((train_x, train_y))
    data_set = data_set.repeat().shuffle(1000).batch(BATCH_SIZE)
    x, y_ = data_set.make_one_shot_iterator().get_next()

    # test_x, and test_y is already the point from the same sequence
    test_x, test_y = generate_data(test_seq_y)
    test_data_set = tf.data.Dataset.from_tensor_slices((test_x, test_y))
    data_set = test_data_set.batch(1)
    test_x_point, test_y_point = data_set.make_one_shot_iterator().get_next()

    with tf.variable_scope("model"):
        _, loss, train_op = lstm_model(x, y_, is_training=True)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(TRAINING_STEPS):
            _, l = sess.run([train_op, loss])  # todo: group

            if i % 100 == 0:
                print("%s loss: %s" % (i, l))

        # predict with trained model
        with tf.variable_scope("model", reuse=True):
            prediction, _, _ = lstm_model(
                test_x_point, [0.0], is_training=False)

            predictions = []
            labels = []
            # how to choose test_iter times. todo: think
            for i in range(1600):
                p, l = sess.run([prediction, test_y_point])
                predictions.append(p)
                labels.append(l)

            predictions = np.array(predictions).squeeze()
            labels = np.array(labels).squeeze()

            plt.figure("test_pred_label")
            plt.plot(predictions, label="predictions")
            plt.plot(labels, label="labels")
            plt.legend()
            plt.show()


