from ops import *


def sobel(image_batch):
    sobel_x = tf.constant([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], tf.float32)
    sobel_x_filter = tf.reshape(sobel_x, [3, 3, 1, 1])
    sobel_y = tf.constant([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], tf.float32)
    sobel_y_filter = tf.reshape(sobel_y, [3, 3, 1, 1])
    filtered_x = tf.nn.conv2d(image_batch, sobel_x_filter, strides=[1, 1, 1, 1], padding='SAME')
    filtered_y = tf.nn.conv2d(image_batch, sobel_y_filter, strides=[1, 1, 1, 1], padding='SAME')
    fileterd_xy = tf.sqrt(tf.square(filtered_x) + tf.square(filtered_y))
    return fileterd_xy


def sobel2(image_batch):
    sobel_x = tf.constant([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], tf.float32)
    sobel_x_filter = tf.reshape(sobel_x, [3, 3, 1, 1])
    sobel_y = tf.constant([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], tf.float32)
    sobel_y_filter = tf.reshape(sobel_y, [3, 3, 1, 1])
    filtered_x = tf.nn.conv2d(image_batch, sobel_x_filter, strides=[1, 1, 1, 1], padding='SAME')
    filtered_y = tf.nn.conv2d(image_batch, sobel_y_filter, strides=[1, 1, 1, 1], padding='SAME')
    # fileterd_xy = tf.sqrt(tf.square(filtered_x) + tf.square(filtered_y))
    return [filtered_x, filtered_y]


def get_sobel_batch(image_batch):
    assert image_batch.shape[-1].value == 2
    batch_size = image_batch.shape[0].value
    height = image_batch.shape[1].value
    width = image_batch.shape[2].value

    batch1 = image_batch[:, :, :, 0]
    batch2 = image_batch[:, :, :, 1]
    batch = tf.reshape(tf.concat([batch1, batch2], 0), [batch_size * 2, height, width, 1])
    batch_sobel = sobel2(batch)
    batch_sobel = tf.concat([batch_sobel[0], batch_sobel[1]], 0)

    return batch_sobel


# the loss of colorization network
def loss_colorization(out_ab_batch, color_ab_batch, index_ab_batch, l1, l2):
    with tf.name_scope('loss_colorization'):
        # Lgt: the ground truth loss
        loss1 = tf.losses.huber_loss(out_ab_batch, color_ab_batch) * l1
        # Lgl: the difference of color themes between the generated ab channels and the input color themes
        loss2 = tf.losses.huber_loss(out_ab_batch, index_ab_batch) * l2
        # Lst: the gradient loss
        gra_out_ab = get_sobel_batch(out_ab_batch)
        gra_color_ab = get_sobel_batch(color_ab_batch)
        loss3 = tf.reduce_mean(tf.square(gra_out_ab - gra_color_ab)) * 10

        loss_total = loss1 + loss2 + loss3
        return loss_total, [loss1, loss2, loss3]


# the loss of residual network
def loss_residual(out_ab_batch, color_ab_batch):
    with tf.name_scope('losses_residual'):
        loss4 = tf.reduce_mean(tf.abs(out_ab_batch - color_ab_batch))
        return loss4, [tf.constant(0.0), tf.constant(0.0), tf.constant(0.0), tf.constant(0.0)]


def training(loss, global_step, start_learning_rate, decay_steps, decay_rate, var_list):
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        lr = tf.train.exponential_decay(start_learning_rate, global_step, decay_steps, decay_rate, staircase=True)
        optimizer = tf.train.AdamOptimizer(lr)
        train_op = optimizer.minimize(loss, global_step=global_step, var_list=var_list)

    return train_op, lr


# colorization network without gradient
def inference3(img_l_batch, theme_ab_batch, theme_mask_batch, local_ab_batch, local_mask_batch,
                 is_training=True, scope_name='UG'):
    """
    :param img_l_batch:     the L-channel of input images, [N, H, W, 1]
    :param local_ab_batch:  the ab-channel of local points, [N, H, W, 2]
    :param local_mask_batch:    the ab-channel of local points mask, [N, H, W, 1]
    :param scope_name:      name
    :return: the ab-channel of output images
    """
    assert img_l_batch.get_shape()[-1] == 1
    assert theme_ab_batch.get_shape()[-1] == 2
    assert theme_mask_batch.get_shape()[-1] == 1
    assert local_ab_batch.get_shape()[-1] == 2
    assert local_mask_batch.get_shape()[-1] == 1

    print('==========', scope_name, '==========')
    ngf = 64
    theme_batch = tf.concat([theme_ab_batch, theme_mask_batch], axis=3)
    local_batch = tf.concat([local_ab_batch, local_mask_batch], axis=3)
    print('Img Inputs:', img_l_batch)
    print('Theme Inputs:', theme_batch)
    print('Local Inputs:', local_batch)
    print()

    with tf.variable_scope(scope_name, reuse=tf.AUTO_REUSE):
        print('ThemeBlock')
        theme_batch = tf.reshape(theme_batch, [img_l_batch.get_shape()[0], 1, 1, -1])
        print(theme_batch)
        glob_conv1 = conv2d(theme_batch, ngf * 8, 1, 1, activation_fn=tf.nn.relu,
                            norm_fn=tf.layers.batch_normalization, is_training=is_training, scope_name='glob_conv1')
        print(glob_conv1)
        glob_conv2 = conv2d(glob_conv1, ngf * 8, 1, 1, activation_fn=tf.nn.relu,
                            norm_fn=tf.layers.batch_normalization, is_training=is_training, scope_name='glob_conv2')
        print(glob_conv2)
        glob_conv3 = conv2d(glob_conv2, ngf * 8, 1, 1, activation_fn=tf.nn.relu,
                            norm_fn=tf.layers.batch_normalization, is_training=is_training, scope_name='glob_conv3')
        print(glob_conv3)
        glob_conv4 = conv2d(glob_conv3, ngf * 8, 1, 1, activation_fn=tf.nn.relu,
                            norm_fn=tf.layers.batch_normalization, is_training=is_training, scope_name='glob_conv4')
        print(glob_conv4, end='\n\n')

        print('LocalBlock')
        ab_conv1_1 = conv2d(local_batch, ngf, 3, 1, activation_fn=tf.nn.relu,
                            norm_fn=None, is_training=is_training, scope_name='ab_conv1_1')
        print(ab_conv1_1)
        bw_conv1_1 = conv2d(img_l_batch, ngf, 3, 1, activation_fn=tf.nn.relu,
                            norm_fn=None, is_training=is_training, scope_name='bw_conv1_1')
        print(bw_conv1_1)

        print('ConvBlock 1')
        conv1_1 = ab_conv1_1 + bw_conv1_1  # TODO: add the local inputs
        conv1_1 = conv2d(conv1_1, ngf, 3, 1, activation_fn=tf.nn.relu,
                         norm_fn=None, is_training=is_training, scope_name='conv1_1')
        print(conv1_1)
        conv1_2 = conv2d(conv1_1, ngf, 3, 1, activation_fn=tf.nn.relu,
                         norm_fn=tf.layers.batch_normalization, is_training=is_training, scope_name='conv1_2')
        print(conv1_2)
        conv1_2_ss = depth_wise_conv2d(conv1_2, 1, 1, 2, activation_fn=None, scope_name='conv1_2_ss')
        print(conv1_2_ss, end='\n\n')

        print('ConvBlock 2')
        conv2_1 = conv2d(conv1_2_ss, ngf * 2, 3, 1, activation_fn=tf.nn.relu,
                         norm_fn=None, is_training=is_training, scope_name='conv2_1')
        print(conv2_1)
        conv2_2 = conv2d(conv2_1, ngf * 2, 3, 1, activation_fn=tf.nn.relu,
                         norm_fn=tf.layers.batch_normalization, is_training=is_training, scope_name='conv2_2')
        print(conv2_2)
        conv2_2_ss = depth_wise_conv2d(conv2_2, 1, 1, 2, activation_fn=None, scope_name='conv2_2_ss')
        print(conv2_2_ss, end='\n\n')

        print('ConvBlock 3')
        conv3_1 = conv2d(conv2_2_ss, ngf * 4, 3, 1, activation_fn=tf.nn.relu,
                         norm_fn=None, is_training=is_training, scope_name='conv3_1')
        print(conv3_1)
        conv3_2 = conv2d(conv3_1, ngf * 4, 3, 1, activation_fn=tf.nn.relu,
                         norm_fn=None, is_training=is_training, scope_name='conv3_2')
        print(conv3_2)
        conv3_3 = conv2d(conv3_2, ngf * 4, 3, 1, activation_fn=tf.nn.relu,
                         norm_fn=tf.layers.batch_normalization, is_training=is_training, scope_name='conv3_3')
        print(conv3_3)
        conv3_3_ss = depth_wise_conv2d(conv3_3, 1, 1, 2, activation_fn=None, scope_name='conv3_3_ss')
        print(conv3_3_ss, end='\n\n')

        print('ConvBlock 4')
        conv4_1 = conv2d(conv3_3_ss, ngf * 8, 3, 1, activation_fn=tf.nn.relu,
                         norm_fn=None, is_training=is_training, scope_name='conv4_1')
        print(conv4_1)
        conv4_2 = conv2d(conv4_1, ngf * 8, 3, 1, activation_fn=tf.nn.relu,
                         norm_fn=None, is_training=is_training, scope_name='conv4_2')
        print(conv4_2)
        conv4_3 = conv2d(conv4_2, ngf * 8, 3, 1, activation_fn=tf.nn.relu,
                         norm_fn=tf.layers.batch_normalization, is_training=is_training, scope_name='conv4_3')
        print(conv4_3, end='\n\n')

        print('ConvBlock 5')
        conv4_3 = conv4_3 + glob_conv4  # TODO: add the global inputs
        conv5_1 = conv2d(conv4_3, ngf * 8, 3, 1, dilation=2, activation_fn=tf.nn.relu,
                         norm_fn=None, is_training=is_training, scope_name='conv5_1')
        print(conv5_1)
        conv5_2 = conv2d(conv5_1, ngf * 8, 3, 1, dilation=2, activation_fn=tf.nn.relu,
                         norm_fn=None, is_training=is_training, scope_name='conv5_2')
        print(conv5_2)
        conv5_3 = conv2d(conv5_2, ngf * 8, 3, 1, dilation=2, activation_fn=tf.nn.relu,
                         norm_fn=tf.layers.batch_normalization, is_training=is_training, scope_name='conv5_3')
        print(conv5_3, end='\n\n')

        print('ConvBlock 6')
        conv6_1 = conv2d(conv5_3, ngf * 8, 3, 1, dilation=2, activation_fn=tf.nn.relu,
                         norm_fn=None, is_training=is_training, scope_name='conv6_1')
        print(conv6_1)
        conv6_2 = conv2d(conv6_1, ngf * 8, 3, 1, dilation=2, activation_fn=tf.nn.relu,
                         norm_fn=None, is_training=is_training, scope_name='conv6_2')
        print(conv6_2)
        conv6_3 = conv2d(conv6_2, ngf * 8, 3, 1, dilation=2, activation_fn=tf.nn.relu,
                         norm_fn=tf.layers.batch_normalization, is_training=is_training, scope_name='conv6_3')
        print(conv6_3, end='\n\n')

        print('ConvBlock 7')
        conv7_1 = conv2d(conv6_3, ngf * 8, 3, 1, activation_fn=tf.nn.relu,
                         norm_fn=None, is_training=is_training, scope_name='conv7_1')
        print(conv7_1)
        conv7_2 = conv2d(conv7_1, ngf * 8, 3, 1, activation_fn=tf.nn.relu,
                         norm_fn=None, is_training=is_training, scope_name='conv7_2')
        print(conv7_2)
        conv7_3 = conv2d(conv7_2, ngf * 8, 3, 1, activation_fn=tf.nn.relu,
                         norm_fn=tf.layers.batch_normalization, is_training=is_training, scope_name='conv7_3')
        print(conv7_3, end='\n\n')

        print('ConvBlock 8')
        conv3_3_short = conv2d(conv3_3, ngf * 4, 3, 1, activation_fn=None,
                               is_training=is_training, scope_name='conv3_3_short')
        conv8_1 = conv2d_transpose(conv7_3, ngf * 4, 4, 2, activation_fn=None,
                                   is_training=is_training, scope_name='conv8_1')
        conv8_1_comb = tf.nn.relu(conv3_3_short + conv8_1)
        print(conv8_1_comb)
        conv8_2 = conv2d(conv8_1_comb, ngf * 4, 3, 1, activation_fn=tf.nn.relu,
                         norm_fn=None, is_training=is_training, scope_name='conv8_2')
        print(conv8_2)
        conv8_3 = conv2d(conv8_2, ngf * 4, 3, 1, activation_fn=tf.nn.relu,
                         norm_fn=tf.layers.batch_normalization, is_training=is_training, scope_name='conv8_3')
        print(conv8_3, end='\n\n')

        print('ConvBlock 9')
        conv2_2_short = conv2d(conv2_2, ngf * 2, 3, 1, activation_fn=None,
                               is_training=is_training, scope_name='conv2_2_short')
        conv9_1 = conv2d_transpose(conv8_3, ngf * 2, 4, 2, activation_fn=None,
                                   is_training=is_training, scope_name='conv9_1')
        conv9_1_comb = tf.nn.relu(conv2_2_short + conv9_1)
        print(conv9_1_comb)
        conv9_2 = conv2d(conv9_1_comb, ngf * 2, 3, 1, activation_fn=tf.nn.relu,
                         norm_fn=tf.layers.batch_normalization, is_training=is_training, scope_name='conv9_2')
        print(conv9_2, end='\n\n')

        print('ConvBlock 10')
        conv1_2_short = conv2d(conv1_2, ngf * 2, 3, 1, activation_fn=None,
                               is_training=is_training, scope_name='conv1_2_short')
        conv10_1 = conv2d_transpose(conv9_2, ngf * 2, 4, 2, activation_fn=None,
                                    is_training=is_training, scope_name='conv10_1')
        conv10_1_comb = tf.nn.relu(conv1_2_short + conv10_1)
        print(conv10_1_comb)
        conv10_2 = conv2d(conv10_1_comb, ngf * 2, 3, 1, activation_fn=tf.nn.relu,
                          norm_fn=tf.layers.batch_normalization, is_training=is_training, scope_name='conv10_2')
        print(conv10_2, end='\n\n')

        print('OutputBlock')
        conv10_ab = conv2d(conv10_2, 2, 1, 1, activation_fn=tf.nn.tanh,
                           norm_fn=None, is_training=is_training, scope_name='conv10_ab')
        print(conv10_ab, end='\n\n')

    return conv10_ab


# colorization network
def inference3_1(img_l_batch, img_l_gra_batch, theme_ab_batch, theme_mask_batch, local_ab_batch, local_mask_batch,
                 is_training=True, scope_name='UG'):
    """
    :param img_l_batch:     the L-channel of input images, [N, H, W, 1]
    :param local_ab_batch:  the ab-channel of local points, [N, H, W, 2]
    :param local_mask_batch:    the ab-channel of local points mask, [N, H, W, 1]
    :param scope_name:      name
    :return: the ab-channel of output images
    """
    assert img_l_batch.get_shape()[-1] == 1
    assert img_l_gra_batch.get_shape()[-1] == 1
    assert theme_ab_batch.get_shape()[-1] == 2
    assert theme_mask_batch.get_shape()[-1] == 1
    assert local_ab_batch.get_shape()[-1] == 2
    assert local_mask_batch.get_shape()[-1] == 1

    print('==========', scope_name, '==========')
    ngf = 64
    theme_batch = tf.concat([theme_ab_batch, theme_mask_batch], axis=3)
    local_batch = tf.concat([local_ab_batch, local_mask_batch], axis=3)
    print('Img Inputs:', img_l_batch)
    print('Theme Inputs:', theme_batch)
    print('Local Inputs:', local_batch)
    print()

    with tf.variable_scope(scope_name, reuse=tf.AUTO_REUSE):
        print('ThemeBlock')
        theme_batch = tf.reshape(theme_batch, [img_l_batch.get_shape()[0], 1, 1, -1])
        print(theme_batch)
        glob_conv1 = conv2d(theme_batch, ngf * 8, 1, 1, activation_fn=tf.nn.relu,
                            norm_fn=tf.layers.batch_normalization, is_training=is_training, scope_name='glob_conv1')
        print(glob_conv1)
        glob_conv2 = conv2d(glob_conv1, ngf * 8, 1, 1, activation_fn=tf.nn.relu,
                            norm_fn=tf.layers.batch_normalization, is_training=is_training, scope_name='glob_conv2')
        print(glob_conv2)
        glob_conv3 = conv2d(glob_conv2, ngf * 8, 1, 1, activation_fn=tf.nn.relu,
                            norm_fn=tf.layers.batch_normalization, is_training=is_training, scope_name='glob_conv3')
        print(glob_conv3)
        glob_conv4 = conv2d(glob_conv3, ngf * 8, 1, 1, activation_fn=tf.nn.relu,
                            norm_fn=tf.layers.batch_normalization, is_training=is_training, scope_name='glob_conv4')
        print(glob_conv4, end='\n\n')

        print('LocalBlock')
        ab_conv1_1 = conv2d(local_batch, ngf, 3, 1, activation_fn=tf.nn.relu,
                            norm_fn=None, is_training=is_training, scope_name='ab_conv1_1')
        print(ab_conv1_1)
        bw_conv1_1 = conv2d(img_l_batch, ngf, 3, 1, activation_fn=tf.nn.relu,
                            norm_fn=None, is_training=is_training, scope_name='bw_conv1_1')
        print(bw_conv1_1)
        gra_conv1_1 = conv2d(img_l_gra_batch, ngf, 3, 1, activation_fn=tf.nn.relu,
                             norm_fn=None, is_training=is_training, scope_name='gra_conv1_1')
        print(gra_conv1_1, end='\n\n')

        print('ConvBlock 1')
        conv1_1 = ab_conv1_1 + bw_conv1_1 + gra_conv1_1  # TODO: add the local inputs
        conv1_1 = conv2d(conv1_1, ngf, 3, 1, activation_fn=tf.nn.relu,
                         norm_fn=None, is_training=is_training, scope_name='conv1_1')
        print(conv1_1)
        conv1_2 = conv2d(conv1_1, ngf, 3, 1, activation_fn=tf.nn.relu,
                         norm_fn=tf.layers.batch_normalization, is_training=is_training, scope_name='conv1_2')
        print(conv1_2)
        conv1_2_ss = depth_wise_conv2d(conv1_2, 1, 1, 2, activation_fn=None, scope_name='conv1_2_ss')
        print(conv1_2_ss, end='\n\n')

        print('ConvBlock 2')
        conv2_1 = conv2d(conv1_2_ss, ngf * 2, 3, 1, activation_fn=tf.nn.relu,
                         norm_fn=None, is_training=is_training, scope_name='conv2_1')
        print(conv2_1)
        conv2_2 = conv2d(conv2_1, ngf * 2, 3, 1, activation_fn=tf.nn.relu,
                         norm_fn=tf.layers.batch_normalization, is_training=is_training, scope_name='conv2_2')
        print(conv2_2)
        conv2_2_ss = depth_wise_conv2d(conv2_2, 1, 1, 2, activation_fn=None, scope_name='conv2_2_ss')
        print(conv2_2_ss, end='\n\n')

        print('ConvBlock 3')
        conv3_1 = conv2d(conv2_2_ss, ngf * 4, 3, 1, activation_fn=tf.nn.relu,
                         norm_fn=None, is_training=is_training, scope_name='conv3_1')
        print(conv3_1)
        conv3_2 = conv2d(conv3_1, ngf * 4, 3, 1, activation_fn=tf.nn.relu,
                         norm_fn=None, is_training=is_training, scope_name='conv3_2')
        print(conv3_2)
        conv3_3 = conv2d(conv3_2, ngf * 4, 3, 1, activation_fn=tf.nn.relu,
                         norm_fn=tf.layers.batch_normalization, is_training=is_training, scope_name='conv3_3')
        print(conv3_3)
        conv3_3_ss = depth_wise_conv2d(conv3_3, 1, 1, 2, activation_fn=None, scope_name='conv3_3_ss')
        print(conv3_3_ss, end='\n\n')

        print('ConvBlock 4')
        conv4_1 = conv2d(conv3_3_ss, ngf * 8, 3, 1, activation_fn=tf.nn.relu,
                         norm_fn=None, is_training=is_training, scope_name='conv4_1')
        print(conv4_1)
        conv4_2 = conv2d(conv4_1, ngf * 8, 3, 1, activation_fn=tf.nn.relu,
                         norm_fn=None, is_training=is_training, scope_name='conv4_2')
        print(conv4_2)
        conv4_3 = conv2d(conv4_2, ngf * 8, 3, 1, activation_fn=tf.nn.relu,
                         norm_fn=tf.layers.batch_normalization, is_training=is_training, scope_name='conv4_3')
        print(conv4_3, end='\n\n')

        print('ConvBlock 5')
        conv4_3 = conv4_3 + glob_conv4  # TODO: add the global inputs
        conv5_1 = conv2d(conv4_3, ngf * 8, 3, 1, dilation=2, activation_fn=tf.nn.relu,
                         norm_fn=None, is_training=is_training, scope_name='conv5_1')
        print(conv5_1)
        conv5_2 = conv2d(conv5_1, ngf * 8, 3, 1, dilation=2, activation_fn=tf.nn.relu,
                         norm_fn=None, is_training=is_training, scope_name='conv5_2')
        print(conv5_2)
        conv5_3 = conv2d(conv5_2, ngf * 8, 3, 1, dilation=2, activation_fn=tf.nn.relu,
                         norm_fn=tf.layers.batch_normalization, is_training=is_training, scope_name='conv5_3')
        print(conv5_3, end='\n\n')

        print('ConvBlock 6')
        conv6_1 = conv2d(conv5_3, ngf * 8, 3, 1, dilation=2, activation_fn=tf.nn.relu,
                         norm_fn=None, is_training=is_training, scope_name='conv6_1')
        print(conv6_1)
        conv6_2 = conv2d(conv6_1, ngf * 8, 3, 1, dilation=2, activation_fn=tf.nn.relu,
                         norm_fn=None, is_training=is_training, scope_name='conv6_2')
        print(conv6_2)
        conv6_3 = conv2d(conv6_2, ngf * 8, 3, 1, dilation=2, activation_fn=tf.nn.relu,
                         norm_fn=tf.layers.batch_normalization, is_training=is_training, scope_name='conv6_3')
        print(conv6_3, end='\n\n')

        print('ConvBlock 7')
        conv7_1 = conv2d(conv6_3, ngf * 8, 3, 1, activation_fn=tf.nn.relu,
                         norm_fn=None, is_training=is_training, scope_name='conv7_1')
        print(conv7_1)
        conv7_2 = conv2d(conv7_1, ngf * 8, 3, 1, activation_fn=tf.nn.relu,
                         norm_fn=None, is_training=is_training, scope_name='conv7_2')
        print(conv7_2)
        conv7_3 = conv2d(conv7_2, ngf * 8, 3, 1, activation_fn=tf.nn.relu,
                         norm_fn=tf.layers.batch_normalization, is_training=is_training, scope_name='conv7_3')
        print(conv7_3, end='\n\n')

        print('ConvBlock 8')
        conv3_3_short = conv2d(conv3_3, ngf * 4, 3, 1, activation_fn=None,
                               is_training=is_training, scope_name='conv3_3_short')
        conv8_1 = conv2d_transpose(conv7_3, ngf * 4, 4, 2, activation_fn=None,
                                   is_training=is_training, scope_name='conv8_1')
        conv8_1_comb = tf.nn.relu(conv3_3_short + conv8_1)
        print(conv8_1_comb)
        conv8_2 = conv2d(conv8_1_comb, ngf * 4, 3, 1, activation_fn=tf.nn.relu,
                         norm_fn=None, is_training=is_training, scope_name='conv8_2')
        print(conv8_2)
        conv8_3 = conv2d(conv8_2, ngf * 4, 3, 1, activation_fn=tf.nn.relu,
                         norm_fn=tf.layers.batch_normalization, is_training=is_training, scope_name='conv8_3')
        print(conv8_3, end='\n\n')

        print('ConvBlock 9')
        conv2_2_short = conv2d(conv2_2, ngf * 2, 3, 1, activation_fn=None,
                               is_training=is_training, scope_name='conv2_2_short')
        conv9_1 = conv2d_transpose(conv8_3, ngf * 2, 4, 2, activation_fn=None,
                                   is_training=is_training, scope_name='conv9_1')
        conv9_1_comb = tf.nn.relu(conv2_2_short + conv9_1)
        print(conv9_1_comb)
        conv9_2 = conv2d(conv9_1_comb, ngf * 2, 3, 1, activation_fn=tf.nn.relu,
                         norm_fn=tf.layers.batch_normalization, is_training=is_training, scope_name='conv9_2')
        print(conv9_2, end='\n\n')

        print('ConvBlock 10')
        conv1_2_short = conv2d(conv1_2, ngf * 2, 3, 1, activation_fn=None,
                               is_training=is_training, scope_name='conv1_2_short')
        conv10_1 = conv2d_transpose(conv9_2, ngf * 2, 4, 2, activation_fn=None,
                                    is_training=is_training, scope_name='conv10_1')
        conv10_1_comb = tf.nn.relu(conv1_2_short + conv10_1)
        print(conv10_1_comb)
        conv10_2 = conv2d(conv10_1_comb, ngf * 2, 3, 1, activation_fn=tf.nn.relu,
                          norm_fn=tf.layers.batch_normalization, is_training=is_training, scope_name='conv10_2')
        print(conv10_2, end='\n\n')

        print('OutputBlock')
        conv10_ab = conv2d(conv10_2, 2, 1, 1, activation_fn=tf.nn.tanh,
                           norm_fn=None, is_training=is_training, scope_name='conv10_ab')
        print(conv10_ab, end='\n\n')

    return conv10_ab


def Conv2d(batch_input, n_fiter, filter_size, strides, act=None, padding='SAME', name='conv'):
    with tf.variable_scope(name):
        in_channels = batch_input.get_shape()[3]
        filters = tf.get_variable('filter', [filter_size, filter_size, in_channels, n_fiter], dtype=tf.float32,
                                  initializer=tf.random_normal_initializer(0, 0.02))
        conv = tf.nn.conv2d(batch_input, filters, [1, strides, strides, 1], padding=padding)
        if act is not None:
            conv = act(conv)
        return conv


def Elementwise(n1, n2, act, name):
    with tf.variable_scope(name):
        return act(n1, n2)


# residual network
def gen_PRLNet(content_map, inputs, out_channels, scope_name='PRLNet'):
    # content_map = content_map * 2. - 1
    # inputs = inputs * 2. - 1

    with tf.variable_scope(scope_name, reuse=tf.AUTO_REUSE):
        # ------------------------------ Detail Generation ------------------------------
        fussion = tf.concat([content_map, inputs], axis=3)
        n = Conv2d(fussion, 64, filter_size=3, strides=1, padding='SAME', name='n64s1/c')

        for i in range(8):
            nn = Conv2d(n, 64, filter_size=3, strides=1, act=tf.nn.relu, padding='SAME', name='dn64s1/c1/%s' % i)
            nn = Conv2d(nn, 64, filter_size=3, strides=1, padding='SAME', name='dn64s1/c2/%s' % i)
            nn = Elementwise(n, nn, tf.add, 'db_residual_add/%s' % i)
            n = nn

        n = Conv2d(n, 256, filter_size=3, strides=1, act=None, padding='SAME', name='n256s1/2')
        n = Conv2d(n, out_channels, filter_size=1, strides=1, padding='SAME', name='out')
        # detail_map = tf.nn.tanh(n)
        detail_map = n

        output_map = tf.add(content_map, detail_map)
        output_map = tf.nn.tanh(output_map)

        return detail_map, output_map
