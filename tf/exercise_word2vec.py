#!/usr/bin/python
# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf

from collections import Counter


l_rate = 0.01
n_sample = 100
embedding_size = 200

inputs = tf.placeholder(tf.int32, [None], "input")
labels = tf.placeholder(tf.int32, [None, None], "labels")

embedding = tf.Variable(
    tf.random_uniform([vocabulary_size, embedding_size], -1, -1))
embed = tf.nn.embedding_lookup(embedding, inputs)



def read_file(filename, size=100000000):
    """
    read a file
    :param filename:
    :param size: the block size to read
    :return:
    """
    with open(filename, "r") as f:
        res = f.read(size)
    return res


def preprocessor(text_str, freq=5):
    """
    pre processor, delete words that have a very low frequency
    :param text_str:
    :param freq:
    :return:
    """
    text_str = text_str.lower()
    text_str = text_str.replace('.', ' <PERIOD> ')
    text_str = text_str.replace(',', ' <COMMA> ')
    text_str = text_str.replace('"', ' <QUOTATION_MARK> ')
    text_str = text_str.replace(';', ' <SEMICOLON> ')
    text_str = text_str.replace('!', ' <EXCLAMATION_MARK> ')
    text_str = text_str.replace('?', ' <QUESTION_MARK> ')
    text_str = text_str.replace('(', ' <LEFT_PAREN> ')
    text_str = text_str.replace(')', ' <RIGHT_PAREN> ')
    text_str = text_str.replace('--', ' <HYPHENS> ')
    text_str = text_str.replace('?', ' <QUESTION_MARK> ')
    # text_str = text_str.replace('\n', ' <NEW_LINE> ')
    text_str = text_str.replace(':', ' <COLON> ')
    words = text_str.split()

    words_count = Counter(words)
    # delete words whose frequency is too low
    trimmed_words = [word for word in words if words_count[word] > freq]

    return trimmed_words


def get_target(words, idx, window_size):
    """
    get context of words with the idx of word
    :param words: list of word
    :param idx: index of input word
    :param window_size: size of window
    :return:
    """
    start_point = idx - window_size if (idx - window_size) > 0 else 0
    # when slice a list with a[start: end], if end > len(a), it raise no error
    end_point = idx + window_size

    res = set(words[start_point: idx]).union(set(words[idx + 1: end_point]))
    return list(res)


def get_batches(words, batch_size, window_size=5):
    """
    generator of input_word, output_word pair
    one input_word to many output_word
    :param words: list of words
    :param batch_size:
    :param window_size:
    :return:
    """
    batch_num = len(words) // batch_size
    # drop not used item at the tail
    words_tmp = words[:batch_num * batch_size]

    # lines above is not too important
    for idx in range(0, len(words_tmp), batch_size):
        x = []
        y = []
        # get a batch
        batch = words_tmp[idx: idx + batch_size]
        # get the (input_word, output_word) pair
        # get input word and get its context
        for i in range(len(batch)):
            batch_item_x = batch[i]
            # this is to get the context of x
            batch_y = get_target(batch, i, window_size)
            # align length
            x.extend([batch_item_x]*len(batch_y))
            y.extend(batch_y)

        yield x, y


def train_words(words, vocabulary_int):
    """
    drop words whose frequency is higher than threshold
    :return:
    """
    words_int = [vocabulary_int[word] for word in words]

    t = 1e-5
    threshold = 0.8

    words_int_count = Counter(words_int)
    total_count = len(words_int)
    # frequency of each word in type of dict
    word_freq = {w: c / float(total_count) for w, c in words_int_count.items()}
    # drop words whose frequency is higher than threshold
    drop_prob = {w: 1 - np.sqrt(t / word_freq[w]) for w in words_int_count}

    train_word = [w for w in words_int if drop_prob[w] < threshold]
    print("number of train_words: %s" % len(train_word))
    return train_word


def vocabulary_dict(words):
    """
    remove duplicate words, and transfer word to int in type of dict
    :param words:
    :return:
    """
    vocabulary = set(words)
    vocabulary_int = {w: c for c, w in enumerate(vocabulary)}
    int_vocabulary = {c: w for c, w in enumerate(vocabulary)}

    print("total words: %s" % len(words))
    print("unique words: %s" % len(vocabulary))
    # transfer words to int numbers
    return vocabulary_int, int_vocabulary


def negative_sample(vocabulary_size, embedding_size):
    softmax_w = tf.Variable(
        tf.truncated_normal([vocabulary_size, embedding_size], stddev=0.1))
    softmax_b = tf.Variable(tf.zeros([vocabulary_size]))

    loss = tf.nn.sampled_softmax_loss(softmax_w, softmax_b,
                                      labels,
                                      embed, n_sample,
                                      vocabulary_size)
    cost = tf.reduce_mean(loss)
    train_step = tf.train.AdamOptimizer(l_rate).minimize(cost)

    return train_step


if __name__ == "__main__":
    file_name = "enwik9"
    text = read_file(file_name, size=100000000)
    words = preprocessor(text)
    print(words[:20])

    vocabulary = set(words)

    vocabulary_int, int_vocabulary = vocabulary_dict(words)
