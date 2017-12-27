#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
from: https://github.com/NELSONZHAO/zhihu/blob/master/skip_gram/Skip-Gram-English-Corpus.ipynb
just for exercise, copyright belong to its original author.
"""
import time
import random

import numpy as np
import tensorflow as tf

from collections import Counter


with open("enwik9", "r") as f:
    text = f.read(100000000)


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

words = preprocessor(text)
print(words[:20])

vocabulary = set(words)
vocabulary_int = {w: c for c, w in enumerate(vocabulary)}
int_vocabulary = {c: w for c, w in enumerate(vocabulary)}

print("total words: %s" % len(words))
print("unique words: %s" % len(vocabulary))
# transfer words to int numbers
words_int = [vocabulary_int[word] for word in words]

t = 1e-5
threshold = 0.8

words_int_count = Counter(words_int)
total_count = len(words_int)

word_freq = {w: c/float(total_count) for w, c in words_int_count.items()}
drop_prob = {w: 1-np.sqrt(t/word_freq[w]) for w in words_int_count}

train_word = [w for w in words_int if drop_prob[w] < threshold]
print(len(train_word))


def get_target(words, idx, window_size):
    """
    get context of words with the idx of word
    :param words: list of word
    :param idx: index of input word
    :param window_size: size of window
    :return:
    """
    start_point = idx - window_size if (idx - window_size) > 0 else 0
    # end_point = idx + window_size if (idx + window_size) < len(words) \
    #     else len(words)
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


train_graph = tf.Graph()
with train_graph.as_default():
    inputs = tf.placeholder(tf.int32, [None], name="inputs")
    labels = tf.placeholder(tf.int32, [None, None], name="labels")

vocabulary_size = len(vocabulary)
embedding_size = 200
with train_graph.as_default():
    embedding = tf.Variable(
        tf.random_uniform([vocabulary_size, embedding_size], -1, -1))
    embed = tf.nn.embedding_lookup(embedding, inputs)

# negative sample
n_sample = 100
with train_graph.as_default():
    softmax_w = tf.Variable(
        tf.truncated_normal([vocabulary_size, embedding_size], stddev=0.01))
    # todo: difference of 2 lines below
    # softmax_b = tf.Variable(tf.zeros([vocabulary_size]))
    softmax_b = tf.Variable(tf.zeros(vocabulary_size))
    # loss
    loss = tf.nn.sampled_softmax_loss(softmax_w, softmax_b,
                                      labels,
                                      embed, n_sample,
                                      vocabulary_size)
    cost = tf.reduce_mean(loss)
    optimizer = tf.train.AdamOptimizer(0.01).minimize(cost)


# validate
with train_graph.as_default():
    valid_size = 16
    valid_window = 100
    # randomly select sample in range of valid_window
    # with size = valid_size / 2
    valid_samples = np.array(random.sample(range(valid_window), valid_size//2))
    valid_samples = np.append(valid_samples,
                              np.array(random.sample(
                                  range(1000, 1000+valid_window),
                                  valid_size//2)))
    valid_size = len(valid_samples)
    valid_data_sets = tf.constant(valid_samples, tf.int32)
    norm = tf.sqrt(tf.reduce_sum(tf.square(embedding), 1, keep_dims=True))
    normalized_embedding = embedding / norm
    # lookup word vec of word to validate
    valid_embedding = tf.nn.embedding_lookup(normalized_embedding,
                                             valid_data_sets)
    similarity = tf.matmul(valid_embedding, tf.transpose(normalized_embedding))

epochs = 10
batch_size = 1000
window_size = 10

# with train_graph.as_default():
#     saver = tf.train.Saver()

with tf.Session(graph=train_graph) as sess:
    iteration = 1
    loss = 0
    sess.run(tf.global_variables_initializer())

    for e in range(1, epochs + 1):
        batches = get_batches(train_word, batch_size, window_size)
        start = time.time()
        for x, y in batches:
            # x, y is a list
            feed = {inputs: x, labels: np.array(y)[:, None]}

            train_loss, _ = sess.run([cost, optimizer], feed_dict=feed)
            loss += train_loss

            if iteration % 100 == 0:
                end = time.time()
                print("mean_loss: %s" % (loss/100.0,))
                loss = 0
                start = time.time()

            if iteration % 1000 == 0:
                sim = similarity.eval()
                for i in range(valid_size):
                    valid_words = int_vocabulary[valid_samples[i]]
                    top_k = 8
                    # todo, check grammar
                    nearest = (-sim[i, :]).argsort()[1: top_k+1]
                    for k in range(top_k):
                        close_word = int_vocabulary[nearest[k]]
                    print("close word: %s" % close_word)
            iteration += 1

    # save_path = saver.save(sess, "save_test.ckpt")
    # embed_mat = sess.run(normalized_embedding)



