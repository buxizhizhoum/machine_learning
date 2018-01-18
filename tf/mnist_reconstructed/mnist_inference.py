#!/usr/bin/python
# -*- coding: utf-8 -*-

import tensorflow as tf

INPUT_NODE = 784
OUTPUT_NODE = 10
LAYER_1_NODE = 784


def get_weight_variables(shape, regularizer):
    weights = tf.get_variable(
        "weights",
        shape=shape,
        initializer=tf.truncated_normal_initializer(stddev=0.1))

    if regularizer is not None:
        tf.add_to_collection("losses", regularizer(weights))

    return weights


def inference(input_tensor, regularizer):
    with tf.variable_scope("layer_1"):
        weights = get_weight_variables([INPUT_NODE, LAYER_1_NODE], regularizer)
        biases = tf.get_variable(
            "biases",
            shape=[LAYER_1_NODE],
            initializer=tf.constant_initializer(0.01))
        layer_1 = tf.nn.relu(tf.matmul(input_tensor, weights) + biases)

    with tf.variable_scope("layer_2"):
        weights = get_weight_variables(
            [LAYER_1_NODE, OUTPUT_NODE], regularizer)
        biases = tf.get_variable(
            "biases",
            shape=[OUTPUT_NODE],
            initializer=tf.constant_initializer(0.01))
        layer_2 = tf.matmul(layer_1, weights) + biases

    return layer_2



















