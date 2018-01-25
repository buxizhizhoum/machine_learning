#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
box not passed
"""
import matplotlib.pyplot as plt
import tensorflow as tf


filename = "/home/buxizhizhoum/1-Work/Documents/2-Codes/learning/tf/17flowers/jpg/0/image_0001.jpg"
image_raw_data = tf.gfile.FastGFile(filename, "rb").read()

with tf.Session() as sess:
    img_data = tf.image.decode_jpeg(image_raw_data)
    print(img_data.eval())

    # figure_1 = plt.figure("original")
    # plt.imshow(img_data.eval())
    # plt.show()

    # resize
    # resized_img = tf.image.resize_images(img_data, [1000, 1000], method=1)
    # print(resized_img.eval())

    # figure_resize = plt.figure("resize")
    # plt.imshow(resized_img.eval())

    # corp or pad
    # corp_or_pad_img = tf.image.resize_image_with_crop_or_pad(
    #     img_data, 1000, 1000)
    # print(corp_or_pad_img.eval())

    # figure_corp_or_pad = plt.figure("corp_or_pad")
    # plt.imshow(corp_or_pad_img.eval())

    # central crop
    # central_crop_img = tf.image.central_crop(img_data, 0.5)

    # figure_central_crop = plt.figure("central_crop")
    # plt.imshow(central_crop_img.eval())

    # flip
    # flip_u_d = tf.image.flip_up_down(img_data)
    # flip_l_r = tf.image.flip_left_right(img_data)

    # figure_u_d = plt.figure("flip_up_down")
    # plt.imshow(flip_u_d.eval())
    # figure_l_r = plt.figure("flip_l_r")
    # plt.imshow(flip_l_r.eval())

    # random flip
    # flip_u_d = tf.image.random_flip_up_down(img_data)
    # flip_l_r = tf.image.random_flip_left_right(img_data)

    # figure_u_d = plt.figure("flip_up_down")
    # plt.imshow(flip_u_d.eval())
    # figure_l_r = plt.figure("flip_l_r")
    # plt.imshow(flip_l_r.eval())

    # adjust brightness
    # adjusted = tf.image.adjust_brightness(img_data, -0.5)

    # figure_brightness = plt.figure("brightness")
    # plt.imshow(adjusted.eval())

    # random brightness
    # random_brightness = tf.image.random_brightness(img_data, max_delta=0.9)
    # figure_random_brightness = plt.figure("random_brightness")
    # plt.imshow(random_brightness.eval())

    # standardization
    # standard = tf.image.per_image_standardization(img_data)
    # figure_standardization = plt.figure("figure_standardization")
    # matplotlib.modellib.unmode_image(standard.eval())
    # plt.imshow(standard.eval())

    # boxes
    boxes = tf.constant([[[0.05, 0.05, 0.9, 0.7], [0.35, 0.47, 0.5, 0.56]]])
    img_data = tf.image.convert_image_dtype(img_data, tf.float32)
    begin, size, bbox_for_draw = tf.image.sample_distorted_bounding_box(
        tf.shape(img_data), bounding_boxes=boxes)

    batched = tf.expand_dims(
        tf.image.convert_image_dtype(img_data, tf.float32), 0)
    image_with_box = tf.image.draw_bounding_boxes(batched, bbox_for_draw)
    distorted_image = tf.slice(img_data, begin, size)
    figure_boxes = plt.figure("boxes")
    plt.imshow(distorted_image.eval())

    plt.show()

    img_data = tf.image.convert_image_dtype(img_data, dtype=tf.uint8)
    encoded_img = tf.image.encode_jpeg(img_data)
    with tf.gfile.GFile("/home/buxizhizhoum/tf_tmp/test_0.jpeg", "wb") as f:
        f.write(encoded_img.eval())

