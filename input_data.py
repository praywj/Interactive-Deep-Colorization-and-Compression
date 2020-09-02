import time
import pathlib
import tensorflow as tf
import numpy as np
import model


def get_all_paths(root_dir, ext='png'):
    root_dir = pathlib.Path(root_dir)
    file_paths = list(map(str, root_dir.rglob('*.' + ext)))

    return file_paths


def get_train_list(dir_list, name_list, ext_list, shuffle=True):
    train_list = []
    for root_dir, name, ext in zip(dir_list, name_list, ext_list):
        tic = time.time()
        file_paths = get_all_paths(root_dir, ext)
        toc = time.time()
        print('[Type:%s][File nums: %d, Time_cost: %.2fs]' % (name, len(file_paths), toc - tic))
        train_list.append(np.asarray(file_paths))

    if shuffle:
        file_count = len(train_list[0])
        rnd_index = np.arange(file_count)
        np.random.shuffle(rnd_index)
        for i, item in enumerate(train_list):
            train_list[i] = item[rnd_index]

    return tuple(train_list)


# only global + only local + both
def get_batch(train_list, image_size, batch_size, capacity, is_random=True):

    filepath_queue = tf.train.slice_input_producer(train_list, shuffle=False)

    # color
    image_rgb = tf.read_file(filepath_queue[0])
    image_rgb = tf.image.decode_png(image_rgb, channels=3)
    image_rgb = tf.image.resize_images(image_rgb, [image_size, image_size])
    image_rgb = tf.cast(image_rgb, tf.float32) / 255.       # 归一化到[0, 1]

    # color theme
    theme_rgb = tf.read_file(filepath_queue[1])
    theme_rgb = tf.image.decode_png(theme_rgb, channels=3)
    theme_rgb = tf.image.resize_images(theme_rgb, [1, 7])
    theme_rgb = tf.cast(theme_rgb, tf.float32) / 255.

    # color theme mask
    theme_mask = tf.read_file(filepath_queue[2])
    theme_mask = tf.image.decode_png(theme_mask, channels=1)
    theme_mask = tf.image.resize_images(theme_mask, [1, 7])
    theme_mask = tf.cast(theme_mask, tf.float32) / 255.
    theme_mask = tf.reshape(theme_mask[:, :, 0], [1, 7, 1])

    # a K-color map by decoding the color image with its representative colors
    index_rgb = tf.read_file(filepath_queue[3])
    index_rgb = tf.image.decode_png(index_rgb, channels=3)
    index_rgb = tf.image.resize_images(index_rgb, [image_size, image_size])
    index_rgb = tf.cast(index_rgb, tf.float32) / 255.

    # local rgb
    point_rgb = tf.read_file(filepath_queue[4])
    point_rgb = tf.image.decode_png(point_rgb, channels=3)
    point_rgb = tf.image.resize_images(point_rgb, [image_size, image_size])
    point_rgb = tf.cast(point_rgb, tf.float32) / 255.

    # local mask
    point_mask = tf.read_file(filepath_queue[5])
    point_mask = tf.image.decode_png(point_mask, channels=1)
    point_mask = tf.image.resize_images(point_mask, [image_size, image_size])
    point_mask = tf.cast(point_mask, tf.float32) / 255.
    point_mask = tf.reshape(point_mask[:, :, 0], [image_size, image_size, 1])

    # set to zero
    theme_rgb_blank = tf.zeros([1, 7, 3], dtype=tf.float32)
    theme_mask_blank = tf.zeros([1, 7, 1], dtype=tf.float32)
    index_blank = image_rgb
    point_rgb_blank = tf.zeros([image_size, image_size, 3], dtype=tf.float32)
    point_mask_blank = tf.zeros([image_size, image_size, 1], dtype=tf.float32)

    # set to zero by random
    rnd = tf.random_uniform(shape=[1], minval=0, maxval=1, dtype=tf.float32)
    rnd = rnd[0]

    def f1():   # only global
        return theme_rgb, theme_mask, index_rgb, point_rgb_blank, point_mask_blank

    def f2():   # only local
        return theme_rgb_blank, theme_mask_blank, index_blank, point_rgb, point_mask

    def f3():   # both
        return theme_rgb, theme_mask, index_rgb, point_rgb, point_mask

    if is_random is True:
        rate1 = 0.05
        rate2 = 0.05
        flag1 = tf.less(rnd, rate1)
        flag2 = tf.logical_and(tf.greater_equal(rnd, rate1), tf.less(rnd, rate1 + rate2))
        flag3 = tf.greater_equal(rnd, rate1 + rate2)
        theme_rgb, theme_mask, index_rgb, point_rgb, point_mask = \
            tf.case({flag1: f1, flag2: f2, flag3: f3}, exclusive=True)

    if is_random is True:
        image_rgb_batch, theme_rgb_batch, theme_mask_batch, index_rgb_batch, point_rgb_batch, point_mask_batch = \
            tf.train.shuffle_batch([image_rgb, theme_rgb, theme_mask, index_rgb, point_rgb, point_mask],
                                   batch_size=batch_size,
                                   capacity=capacity,
                                   min_after_dequeue=500,
                                   num_threads=4)
    else:
        image_rgb_batch, theme_rgb_batch, theme_mask_batch, index_rgb_batch, point_rgb_batch, point_mask_batch = \
            tf.train.batch([image_rgb, theme_rgb, theme_mask, index_rgb, point_rgb, point_mask],
                           batch_size=1,
                           capacity=capacity,
                           num_threads=1)
    return image_rgb_batch, theme_rgb_batch, theme_mask_batch, index_rgb_batch, point_rgb_batch, point_mask_batch


# Convert from RGB space to LAB space, require normalized RGB input, and unnormalized output LAB, require float32
def rgb_to_lab(image_rgb):
    assert image_rgb.get_shape()[-1] == 3

    rgb_pixels = tf.reshape(image_rgb, [-1, 3])
    # RGB to XYZ
    with tf.name_scope("rgb_to_xyz"):
        linear_mask = tf.cast(rgb_pixels <= 0.04045, dtype=tf.float32)
        expoential_mask = tf.cast(rgb_pixels > 0.04045, dtype=tf.float32)
        rgb_pixels = (rgb_pixels / 12.92) * linear_mask +\
                     (((rgb_pixels + 0.055) / 1.055) ** 2.4) * expoential_mask
        transfer_mat = tf.constant([
            [0.412453, 0.212671, 0.019334],
            [0.357580, 0.715160, 0.119193],
            [0.180423, 0.072169, 0.950227]
        ], dtype=tf.float32)
        xyz_pixels = tf.matmul(rgb_pixels, transfer_mat)

    # XYZ to LAB
    with tf.name_scope("xyz_to_lab"):
        # Standardized D65 white point
        xyz_norm_pixels = tf.multiply(xyz_pixels, [1/0.950456, 1.0, 1/1.088754])
        epsilon = 6/29
        linear_mask = tf.cast(xyz_norm_pixels <= epsilon**3, dtype=tf.float32)
        expoential_mask = tf.cast(xyz_norm_pixels > epsilon**3, dtype=tf.float32)
        f_xyf_pixels = (xyz_norm_pixels / (3 * epsilon**2) + 4/29) * linear_mask +\
                       (xyz_norm_pixels**(1/3)) * expoential_mask
        transfer_mat2 = tf.constant([
            [0.0, 500.0, 0.0],
            [116.0, -500.0, 200.0],
            [0.0, 0.0, -200.0]
        ], dtype=tf.float32)
        lab_pixels = tf.matmul(f_xyf_pixels, transfer_mat2) + tf.constant([-16.0, 0.0, 0.0], dtype=tf.float32)

        image_lab = tf.reshape(lab_pixels, tf.shape(image_rgb))

    return image_lab


# LAB space to RGB space
def lab_to_rgb(image_lab):
    assert image_lab.shape[-1] == 3

    lab_pixels = tf.reshape(image_lab, [-1, 3])
    with tf.name_scope('lab_to_xyz'):
        transfer_mat1 = tf.constant([
            [1/116.0, 1/116.0, 1/116.0],
            [1/500.0, 0.0, 0.0],
            [0.0, 0.0, -1/200.0]
        ], dtype=tf.float32)
        fxfyfz_pixels = tf.matmul(lab_pixels + tf.constant([16.0, 0.0, 0.0]), transfer_mat1)
        epsilon = 6/29
        linear_mask = tf.cast(fxfyfz_pixels <= epsilon, dtype=tf.float32)
        expoential_mask = tf.cast(fxfyfz_pixels > epsilon, dtype=tf.float32)
        xyz_pixels = (3 * epsilon**2 * (fxfyfz_pixels - 4/29)) * linear_mask +\
                     (fxfyfz_pixels **3) * expoential_mask
        xyz_pixels = tf.multiply(xyz_pixels, [0.950456, 1.0, 1.088754])

    with tf.name_scope('xyz_to_rgb'):
        transfer_mat2 = tf.constant([
            [3.2404542, -0.9692660, 0.0556434],
            [-1.5371385, 1.8760108, -0.2040259],
            [-0.4985314, 0.0415560, 1.0572252]
        ])
        rgb_pixels = tf.matmul(xyz_pixels, transfer_mat2)
        rgb_pixels = tf.clip_by_value(rgb_pixels, 0.0, 1.0)
        linear_mask = tf.cast(rgb_pixels <= 0.0031308, dtype=tf.float32)
        expoential_mask = tf.cast(rgb_pixels > 0.0031308, dtype=tf.float32)
        rgb_pixels = rgb_pixels * 12.92 * linear_mask +\
                     ((rgb_pixels ** (1/2.4) * 1.055) - 0.055) * expoential_mask

        image_rgb = tf.reshape(rgb_pixels, tf.shape(image_lab))

    return image_rgb


def get_eval_img(img_path, theme_path, theme_mask_path, point_path, point_mask_path):
    image_rgb = tf.read_file(img_path)
    image_rgb = tf.image.decode_png(image_rgb, channels=3)
    image_rgb = tf.image.resize_images(image_rgb, [256, 256])
    image_rgb = tf.cast(image_rgb, tf.float32) / 255.  # 归一化到[0, 1]
    image_rgb = tf.reshape(image_rgb, [1, 256, 256, 3])

    theme_rgb = tf.read_file(theme_path)
    theme_rgb = tf.image.decode_png(theme_rgb, channels=3)
    theme_rgb = tf.image.resize_images(theme_rgb, [1, 7])
    theme_rgb = tf.cast(theme_rgb, tf.float32) / 255.
    theme_rgb = tf.reshape(theme_rgb, [1, 1, 7, 3])

    theme_mask = tf.read_file(theme_mask_path)
    theme_mask = tf.image.decode_png(theme_mask, channels=1)
    theme_mask = tf.image.resize_images(theme_mask, [1, 7])
    theme_mask = tf.cast(theme_mask, tf.float32) / 255.
    theme_mask = tf.reshape(theme_mask[:, :, 0], [1, 1, 7, 1])

    point_rgb = tf.read_file(point_path)
    point_rgb = tf.image.decode_png(point_rgb, channels=3)
    point_rgb = tf.image.resize_images(point_rgb, [256, 256])
    point_rgb = tf.cast(point_rgb, tf.float32) / 255.
    point_rgb = tf.reshape(point_rgb, [1, 256, 256, 3])

    point_mask = tf.read_file(point_mask_path)
    point_mask = tf.image.decode_png(point_mask, channels=1)
    point_mask = tf.image.resize_images(point_mask, [256, 256])
    point_mask = tf.cast(point_mask, tf.float32) / 255.
    point_mask = tf.reshape(point_mask[:, :, 0], [1, 256, 256, 1])

    # TODO: 颜色空间转换
    image_lab = rgb_to_lab(image_rgb)
    image_l = image_lab[:, :, :, 0] / 100. * 2 - 1  # 归一化到[-1, 1]之间
    image_l = tf.reshape(image_l, [1, 256, 256, 1])
    image_l_gra = model.sobel(image_l)

    theme_lab = rgb_to_lab(theme_rgb)
    theme_ab = (theme_lab[:, :, :, 1:] + 128) / 255. * 2 - 1  # 归一化到[-1, 1]之间

    point_lab = rgb_to_lab(point_rgb)
    point_ab = (point_lab[:, :, :, 1:] + 128) / 255. * 2 - 1  # 归一化到[-1, 1]之间

    return image_l, theme_ab, theme_mask, point_ab, point_mask, image_rgb, image_l_gra