#!/usr/bin/python
# -*- coding: utf-8 -*-

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


def generate_data(seq):
    """
    用TIME_STEPS个点，预测下一个点
    :param seq:
    :return:
    """
    x = []
    y = []
    for i in range(len(seq) - TIME_STEPS):
        # x, 第i个点到i+TIME_STEPS-1个点
        x.append([seq[i: i + TIME_STEPS]])  # 用于预测的点
        # 第i+TIME_STEPS个点，上面的点的预测结果
        y.append([seq[i + TIME_STEPS]])  # 预测的点结果
    # todo: generator?
    return np.array(x, dtype=np.float32), np.array(y, dtype=np.float32)


# generate training data and test data
# test_start = (TRAINING_EXAMPLES + TIME_STEPS) * SAMPLE_GAP
# test_end = test_start + (TEST_EXAMPLES + TIME_STEPS) * SAMPLE_GAP
test_start = TRAINING_EXAMPLES * SAMPLE_GAP
test_end = test_start + TEST_EXAMPLES * SAMPLE_GAP
train_sequence_x = np.linspace(
    0, test_start, TRAINING_EXAMPLES + TIME_STEPS, dtype=np.float32)
test_sequence_x = np.linspace(
    test_start, test_end, TEST_EXAMPLES + TIME_STEPS, dtype=np.float32)
print(train_sequence_x)
print(test_sequence_x)

train_sequence = np.sin(train_sequence_x)
test_sequence = np.sin(test_sequence_x)
train_x, train_y = generate_data(train_sequence)
test_x, test_y = generate_data(test_sequence)

# plt.figure("data set")
# plt.plot(train_sequence_x, train_sequence, label="training")
# plt.plot(test_sequence_x, test_sequence, label="test")
# plt.legend()
# plt.show()


def get_a_cell(lstm_size, keep_prob=1):
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


def evaluation(sess, test_x, test_y):
    # 作用是切分传入Tensor的第一个维度，生成相应的dataset，如果是一维则不切分
    ds = tf.data.Dataset.from_tensor_slices((test_x, test_y))
    ds = ds.batch(1)  # 一次拿一个数据
    # make_one_shot_iterator()实例化一个iterator，get_next()对这个iterator进行
    # 迭代，这样就可以从上面创建的dataset中拿出数据
    x, y_ = ds.make_one_shot_iterator().get_next()

    # get variable
    with tf.variable_scope("model", reuse=True):
        prediction, _, _ = lstm_model(x, [0.0], is_training=False)

    predictions = []
    labels = []
    for i in range(TEST_EXAMPLES):
        p, l = sess.run([prediction, y_])
        predictions.append(p)
        labels.append(l)

    predictions = np.array(predictions).squeeze()
    labels = np.array(labels).squeeze()
    rsme = np.sqrt((predictions - labels) ** 2).mean(axis=0)
    print("Mean Square Error: %s" % rsme)

    plt.figure()
    plt.plot(predictions, label="predictions")
    plt.plot(labels, label="labels")
    plt.legend()
    plt.show()


ds = tf.data.Dataset.from_tensor_slices((train_x, train_y))
# 调用repeat()序列无限重复，不会抛出tf.errors.OutOfRangeError异常
ds = ds.repeat().shuffle(1000).batch(BATCH_SIZE)
# ds = ds.batch(1)
x, y_ = ds.make_one_shot_iterator().get_next()

with tf.variable_scope("model"):
    _, loss, train_op = lstm_model(x, y_, True)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print("Evaluate before training")
    evaluation(sess, test_x, test_y)

    for i in range(TRAINING_STEPS):
        _, l = sess.run([train_op, loss])
        if i % 100 == 0:
            print("%s, loss: %s" % (i, l))

    print("Evaluate after training")
    evaluation(sess, test_x, test_y)

