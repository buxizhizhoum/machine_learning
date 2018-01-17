#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
ref: https://github.com/nlintz/TensorFlow-Tutorials/blob/master/08_word2vec.py

skip-gram, predict context according to input words.

at first, there should be a preprocessor for the data, all of the text is
transferred to single words, the words should transferred to int number.
the intention to do this is to meet the input
requirements of tf.nn.embedding_lookup() function.

during the training, the data to feed is in type of {input_word, context_word}

still some questions:
    1. in which step the words are transferred to one hot vectors? or in this
    case, int number is enough?

    2. not very clear about tf.nn.embedding_lookup()
"""
import math
import matplotlib.pyplot as plt
import numpy as np
import random
import tensorflow as tf

from collections import Counter


l_rate = 0.01  # learning_rate
embedding_size = 200  # dimension of vec, if need to visualize, set to 2
num_sampled = 15    # Number of negative examples to sample.
batch_size = 20


sentences = ["the quick brown fox jumped over the lazy dog",
            "I love cats and dogs",
            "we all love cats and dogs",
            "cats and dogs are great",
            "sung likes cats",
            "she loves dogs",
            "cats can be very independent",
            "cats are great companions when they want to be",
            "cats are playful",
            "cats are natural hunters",
            "It's raining cats and dogs",
            "dogs and cats love sung"]


def preprocessor(sentences):
    """
    preprocessor of sentences
    :param sentences: sentence in type of list
    :return:
    """
    text = " ".join(sentences)
    words = text.split(" ")
    # count the appear times of words {word: appear_times}
    word_num_dict = Counter(words)

    return words, word_num_dict

words, word_num_dict = preprocessor(sentences)
# todo extract input, context pair
unique_words = set(words)
# set an id number for each word, each number represent for one word
# 把语料库中的文字整型化,每个整数代表一个单词
word_2_int = {w: c for c, w in enumerate(unique_words)}


def generate_batch(words, size):
    # get random indexes of word in list words
    inputs = []
    labels = []
    random_index = random.sample(range(len(words)), size//2)
    for i in random_index:
        # get one word as input word
        input_tmp = words[i]
        # get context of input word
        if i == 0:
            label_tmp = [words[i+1]]
        elif i == len(words) - 1:
            label_tmp = [words[i-1]]
        else:
            label_tmp = [words[i-1], words[i+1]]

        # transfer input word to int number
        input_int = word_2_int[input_tmp]
        label_int = [word_2_int[word] for word in label_tmp]

        # align format
        for i in range(len(label_int)):
            inputs.append(input_int)  # list
            labels.append([label_int[i]])  # list of list
    return inputs, labels


vocabulary_size = len(unique_words)

# 一个是一组用整型表示的上下文单词，另一个是目标单词
# inputs = tf.placeholder(tf.int32, [batch_size], "input")
# labels = tf.placeholder(tf.int32, [batch_size, 1], "labels")

inputs = tf.placeholder(tf.int32, [None], "input")
labels = tf.placeholder(tf.int32, [None, None], "labels")

# 定义一个嵌套参数矩阵, 用唯一的随机值来初始化这个大矩阵
embeddings = tf.Variable(
    tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))
# input should contain the ids used to look up.
# embeddings are the final word vectors
embed = tf.nn.embedding_lookup(embeddings, inputs)

# 对语料库中的每个单词定义一个权重值和偏差值
nce_weights = tf.Variable(
  tf.truncated_normal([vocabulary_size, embedding_size],
                      stddev=1.0 / math.sqrt(embedding_size)))
nce_biases = tf.Variable(tf.zeros([vocabulary_size]))

loss = tf.reduce_mean(tf.nn.nce_loss(weights=nce_weights,
                                     biases=nce_biases,
                                     labels=labels,
                                     inputs=embed,
                                     num_sampled=num_sampled,
                                     num_classes=vocabulary_size))

# Construct the SGD optimizer using a learning rate of 1.0.
optimizer = tf.train.GradientDescentOptimizer(1.0).minimize(loss)

# Compute the cosine similarity between minibatch examples and all embeddings.
# norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
# normalized_embeddings = embeddings / norm
# valid_embeddings = tf.nn.embedding_lookup(
#   normalized_embeddings, valid_dataset)
# similarity = tf.matmul(
#   valid_embeddings, normalized_embeddings, transpose_b=True)

# Add variable initializer.
init = tf.global_variables_initializer()


if __name__ == "__main__":
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(10000):
            inputs_batch, labels_batch = generate_batch(words, batch_size)
            feed_dict = {inputs: inputs_batch, labels: labels_batch}
            _, cur_loss = sess.run([optimizer, loss], feed_dict=feed_dict)

            if i % 100 == 0:
                print("loss: %s" % cur_loss)

        trained_embeddings = embeddings.eval()
        print(trained_embeddings)

    if trained_embeddings.shape[1] == 2:
        for i, label in enumerate(unique_words):
            x, y = trained_embeddings[i, :]
            plt.scatter(x, y)
            plt.annotate(label, xy=(x, y), xytext=(5, 2),
                         textcoords='offset points', ha='right', va='bottom')
        plt.show()


