import model
import input_data
import matplotlib.pyplot as plt
import time
import os
import tensorflow as tf
import numpy as np


# first step: without residual network
def train1():
    BATCH_SIZE = 16
    CAPACITY = 1000
    MAX_STEP = 300001
    IMAGE_SIZE = 256

    # the parameters to differentiate the influences of three parts
    l1 = 0.9
    l2 = 0.1

    # directory of dataset
    image_color_dir = 'K:\\UserGuide_256\\train\\color_images\\abbey'
    color_map_dir = 'K:\\UserGuide_256\\train\\color_map\\abbey'
    theme_dir = 'K:\\UserGuide_256\\train\\color_theme\\abbey'
    theme_mask_dir = 'K:\\UserGuide_256\\train\\color_theme_mask\\abbey'
    local_dir = 'K:\\UserGuide_256\\train\\local_points\\abbey'
    local_mask_dir = 'K:\\UserGuide_256\\train\\local_points_mask\\abbey'

    # directory of checkpoint
    logs_dir = 'logs_1\\'

    sess = tf.Session()

    # get the training data
    train_list = input_data.get_train_list(
        [image_color_dir, theme_dir, theme_mask_dir, color_map_dir, local_dir, local_mask_dir],
        ['color img', 'theme img', 'theme mask', 'color_map img', 'local img', 'local mask'],
        ['*', '*', 'png', 'png', 'png', 'png'], shuffle=True)

    image_rgb_batch, theme_rgb_batch, theme_mask_batch, index_rgb_batch, point_rgb_batch, point_mask_batch = \
        input_data.get_batch(train_list, IMAGE_SIZE, BATCH_SIZE, CAPACITY, True)

    image_lab_batch = input_data.rgb_to_lab(image_rgb_batch)
    image_l_batch = tf.reshape(image_lab_batch[:, :, :, 0] / 100.0 * 2 - 1, [BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, 1])
    image_l_gra_batch = model.sobel(image_l_batch)
    image_ab_batch = (image_lab_batch[:, :, :, 1:] + 128.) / 255.0 * 2 - 1

    theme_lab_batch = input_data.rgb_to_lab(theme_rgb_batch)
    theme_ab_batch = (theme_lab_batch[:, :, :, 1:] + 128.) / 255.0 * 2 - 1

    index_lab_batch = input_data.rgb_to_lab(index_rgb_batch)
    index_ab_batch = (index_lab_batch[:, :, :, 1:] + 128) / 255.0 * 2 - 1

    point_lab_batch = input_data.rgb_to_lab(point_rgb_batch)
    point_ab_batch = (point_lab_batch[:, :, :, 1:] + 128.) / 255.0 * 2 - 1

    # TODO: training
    out_ab_batch = model.inference3_1(image_l_batch, image_l_gra_batch, theme_ab_batch, theme_mask_batch,
                                      point_ab_batch, point_mask_batch,
                                      is_training=True, scope_name='UserGuide')

    # TODO: envalute
    image_l_test, theme_ab_test, theme_mask_test, point_ab_test, point_mask_test, image_rgb_test, image_gra_test = \
        input_data.get_eval_img('images/img_rgb.png', 'images/theme_rgb.png', 'images/theme_mask.png',
                                   'images/points_rgb.png', 'images/points_mask.png')
    test_ab_out = model.inference3_1(image_l_test, image_gra_test, theme_ab_test, theme_mask_test, point_ab_test,
                                     point_mask_test,
                                     is_training=False, scope_name='UserGuide')
    test_rgb_out = \
        input_data.lab_to_rgb(tf.concat([(image_l_test + 1.) / 2 * 100., (test_ab_out + 1.) / 2 * 255. - 128], axis=3))
    test_psnr = 10 * tf.log(1 / (tf.reduce_mean(tf.square(test_rgb_out - image_rgb_test)))) / np.log(10)

    global_step = tf.train.get_or_create_global_step(sess.graph)
    # compute the loss
    train_loss, loss_paras = model.loss_colorization(out_ab_batch, image_ab_batch, index_ab_batch, l1, l2)

    var_list = tf.trainable_variables()
    paras_count = tf.reduce_sum([tf.reduce_prod(v.shape) for v in var_list])
    print('参数数目:%d' % sess.run(paras_count), end='\n\n')

    train_op, learning_rate = model.training(train_loss, global_step, 1e-3, 4e4, 0.7, var_list)

    saver1 = tf.train.Saver(max_to_keep=10)

    sess.run(tf.global_variables_initializer())  # Variable initialization

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    s_t = time.time()
    try:
        for step in range(MAX_STEP):
            if coord.should_stop():
                print('???')
                break

            _, loss, loss_sub, lr = sess.run([train_op, train_loss, loss_paras, learning_rate])

            if step % 100 == 0:
                runtime = time.time() - s_t
                psnr = sess.run(test_psnr)
                # record the training process
                print('Step: %d, Loss_total: %g, loss1: %g, test_psnr: %.2fdB, learning_rate:%g, '
                      'time:%.2fs, time left: %.2fhours'
                      % (step, loss, loss_sub[0], psnr, lr,
                         runtime, (MAX_STEP - step) * runtime / 360000))
                s_t = time.time()

            if step % 10000 == 0 or step == MAX_STEP - 1:  # save checkpoint
                if not os.path.exists(logs_dir):
                    os.makedirs(logs_dir)
                checkpoint_path = os.path.join(logs_dir, 'model.ckpt')
                saver1.save(sess, checkpoint_path, global_step=step)

            if step % 1000 == 0 or step == MAX_STEP - 1:
                test_out = sess.run(test_rgb_out)
                test_out = test_out[0]

                save_path = 'logs_output/' + logs_dir   # save results
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                save_path += 'step_' + str(step) + '.png'
                plt.imsave(save_path, test_out)

    except tf.errors.OutOfRangeError:
        print('Done.')
    finally:
        coord.request_stop()

    # 等待线程结束
    coord.join(threads=threads)
    sess.close()


# second step: fix the colorization network and train the residual network
def train2():
    BATCH_SIZE = 16
    CAPACITY = 1000
    MAX_STEP = 300001
    IMAGE_SIZE = 256

    # directory of dataset
    image_color_dir = 'K:\\UserGuide_256\\train\\color_images\\abbey'
    color_map_dir = 'K:\\UserGuide_256\\train\\color_map\\abbey'
    theme_dir = 'K:\\UserGuide_256\\train\\color_theme\\abbey'
    theme_mask_dir = 'K:\\UserGuide_256\\train\\color_theme_mask\\abbey'
    local_dir = 'K:\\UserGuide_256\\train\\local_points\\abbey'
    local_mask_dir = 'K:\\UserGuide_256\\train\\local_points_mask\\abbey'

    # directory of checkpoint
    logs_dir = 'logs_2\\'

    sess = tf.Session()

    # get the training data
    train_list = input_data.get_train_list(
        [image_color_dir, theme_dir, theme_mask_dir, color_map_dir, local_dir, local_mask_dir],
        ['color img', 'theme img', 'theme mask', 'color_map img', 'local img', 'local mask'],
        ['*', '*', 'png', 'png', 'png', 'png'], shuffle=True)

    image_rgb_batch, theme_rgb_batch, theme_mask_batch, index_rgb_batch, point_rgb_batch, point_mask_batch = \
        input_data.get_batch(train_list, IMAGE_SIZE, BATCH_SIZE, CAPACITY, True)

    image_lab_batch = input_data.rgb_to_lab(image_rgb_batch)
    image_l_batch = tf.reshape(image_lab_batch[:, :, :, 0] / 100.0 * 2 - 1, [BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, 1])
    image_l_gra_batch = model.sobel(image_l_batch)
    image_ab_batch = (image_lab_batch[:, :, :, 1:] + 128.) / 255.0 * 2 - 1

    theme_lab_batch = input_data.rgb_to_lab(theme_rgb_batch)
    theme_ab_batch = (theme_lab_batch[:, :, :, 1:] + 128.) / 255.0 * 2 - 1

    index_lab_batch = input_data.rgb_to_lab(index_rgb_batch)
    index_ab_batch = (index_lab_batch[:, :, :, 1:] + 128) / 255.0 * 2 - 1

    point_lab_batch = input_data.rgb_to_lab(point_rgb_batch)
    point_ab_batch = (point_lab_batch[:, :, :, 1:] + 128.) / 255.0 * 2 - 1

    # TODO: training
    # colorization network
    out_ab_batch = model.inference3_1(image_l_batch, image_l_gra_batch, theme_ab_batch, theme_mask_batch, point_ab_batch, point_mask_batch,
                                      is_training=False, scope_name='UserGuide')
    # residual network
    _, out_ab_batch2 = model.gen_PRLNet(out_ab_batch, image_l_batch, 2, scope_name='PRLNet')

    # TODO: envalute
    image_l_test, theme_ab_test, theme_mask_test, point_ab_test, point_mask_test, image_rgb_test, image_gra_test = \
        input_data.get_eval_img('images/img_rgb.png', 'images/theme_rgb.png', 'images/theme_mask.png',
                                   'images/points_rgb.png', 'images/points_mask.png')
    test_ab_out = model.inference3_1(image_l_test, image_gra_test, theme_ab_test, theme_mask_test, point_ab_test, point_mask_test,
                                     is_training=False, scope_name='UserGuide')
    _, test_ab_out2 = model.gen_PRLNet(test_ab_out, image_l_test, 2, scope_name='PRLNet')
    test_rgb_out0 = \
        input_data.lab_to_rgb(tf.concat([(image_l_test + 1.) / 2 * 100., (test_ab_out + 1.) / 2 * 255. - 128], axis=3))
    test_rgb_out = \
        input_data.lab_to_rgb(tf.concat([(image_l_test + 1.) / 2 * 100., (test_ab_out2 + 1.) / 2 * 255. - 128], axis=3))
    test_psnr0 = 10 * tf.log(1 / (tf.reduce_mean(tf.square(test_rgb_out0 - image_rgb_test)))) / np.log(10)
    test_psnr = 10 * tf.log(1 / (tf.reduce_mean(tf.square(test_rgb_out - image_rgb_test)))) / np.log(10)

    # 训练残差网络
    var_list = tf.global_variables()
    var_model1 = [var for var in var_list if var.name.startswith('UserGuide')]
    var_model2 = [var for var in var_list if var.name.startswith('PRLNet')]

    paras_count1 = tf.reduce_sum([tf.reduce_prod(v.shape) for v in var_model1])
    paras_count2 = tf.reduce_sum([tf.reduce_prod(v.shape) for v in var_model2])
    print('UserGuide参数数目:%d' % sess.run(paras_count1))
    print('Detailed参数数目:%d' % sess.run(paras_count2))

    global_step = tf.train.get_or_create_global_step(sess.graph)
    train_loss, loss_paras = model.loss_residual(out_ab_batch2, image_ab_batch)
    train_op, learning_rate = model.training(train_loss, global_step, 1e-4, 4e4, 0.7, var_model2)

    saver1 = tf.train.Saver(var_list=var_model1)
    saver2 = tf.train.Saver(var_list=var_model2)

    sess.run(tf.global_variables_initializer())

    print('Loading checkpoint...')
    ckpt = tf.train.get_checkpoint_state('logs_1')
    if ckpt and ckpt.model_checkpoint_path:
        global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
        saver1.restore(sess, ckpt.model_checkpoint_path)
        print('Success, global_step = %s' % global_step)
    else:
        print('Fail')

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    s_t = time.time()
    try:
        for step in range(MAX_STEP):
            if coord.should_stop():
                print('???')
                break

            _, loss, loss_sub, lr = sess.run([train_op, train_loss, loss_paras, learning_rate])

            if step % 100 == 0:
                runtime = time.time() - s_t
                psnr0, psnr = sess.run([test_psnr0, test_psnr])
                print('Step: %d, Loss_total: %g, loss1: %g, test_psnr0: %.2fdB, test_psnr: %.2fdB, learning_rate:%g, '
                      'time:%.2fs, time left: %.2fhours'
                      % (step, loss, loss_sub[0], psnr0, psnr, lr,
                         runtime, (MAX_STEP - step) * runtime / 360000))
                s_t = time.time()

            if step % 10000 == 0 or step == MAX_STEP - 1:
                checkpoint_path = os.path.join(logs_dir, 'model.ckpt')
                saver2.save(sess, checkpoint_path, global_step=step)

            if step % 1000 == 0 or step == MAX_STEP - 1:
                test_out = sess.run(test_rgb_out)
                test_out = test_out[0]

                save_path = 'logs_output/' + logs_dir
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                save_path += 'step_' + str(step) + '.png'
                plt.imsave(save_path, test_out)

    except tf.errors.OutOfRangeError:
        print('Done.')
    finally:
        coord.request_stop()

    # 等待线程结束
    coord.join(threads=threads)
    sess.close()


# jointly train the two networks
def train3():
    BATCH_SIZE = 16
    CAPACITY = 1000
    MAX_STEP = 240001
    IMAGE_SIZE = 256

    # the parameters to differentiate the influences of three parts
    l1 = 0.9
    l2 = 0.1

    # directory of dataset
    image_color_dir = 'K:\\UserGuide_256\\train\\color_images\\abbey'
    color_map_dir = 'K:\\UserGuide_256\\train\\color_map\\abbey'
    theme_dir = 'K:\\UserGuide_256\\train\\color_theme\\abbey'
    theme_mask_dir = 'K:\\UserGuide_256\\train\\color_theme_mask\\abbey'
    local_dir = 'K:\\UserGuide_256\\train\\local_points\\abbey'
    local_mask_dir = 'K:\\UserGuide_256\\train\\local_points_mask\\abbey'

    # directory of checkpoint
    logs_dir = 'logs_3\\'

    sess = tf.Session()

    # get the training data
    train_list = input_data.get_train_list(
        [image_color_dir, theme_dir, theme_mask_dir, color_map_dir, local_dir, local_mask_dir],
        ['color img', 'theme img', 'theme mask', 'color_map img', 'local img', 'local mask'],
        ['*', '*', 'png', 'png', 'png', 'png'], shuffle=True)

    image_rgb_batch, theme_rgb_batch, theme_mask_batch, index_rgb_batch, point_rgb_batch, point_mask_batch = \
        input_data.get_batch(train_list, IMAGE_SIZE, BATCH_SIZE, CAPACITY, True)

    image_lab_batch = input_data.rgb_to_lab(image_rgb_batch)
    image_l_batch = tf.reshape(image_lab_batch[:, :, :, 0] / 100.0 * 2 - 1, [BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, 1])
    image_l_gra_batch = model.sobel(image_l_batch)
    image_ab_batch = (image_lab_batch[:, :, :, 1:] + 128.) / 255.0 * 2 - 1

    theme_lab_batch = input_data.rgb_to_lab(theme_rgb_batch)
    theme_ab_batch = (theme_lab_batch[:, :, :, 1:] + 128.) / 255.0 * 2 - 1

    index_lab_batch = input_data.rgb_to_lab(index_rgb_batch)
    index_ab_batch = (index_lab_batch[:, :, :, 1:] + 128) / 255.0 * 2 - 1

    point_lab_batch = input_data.rgb_to_lab(point_rgb_batch)
    point_ab_batch = (point_lab_batch[:, :, :, 1:] + 128.) / 255.0 * 2 - 1

    # TODO: training
    out_ab_batch = model.inference3_1(image_l_batch, image_l_gra_batch, theme_ab_batch, theme_mask_batch, point_ab_batch, point_mask_batch,
                                      is_training=True, scope_name='UserGuide')
    _, out_ab_batch2 = model.gen_PRLNet(out_ab_batch, image_l_batch, 2, scope_name='PRLNet')

    # TODO: envalute
    image_l_test, theme_ab_test, theme_mask_test, point_ab_test, point_mask_test, image_rgb_test, image_gra_test = \
        input_data.get_eval_img('images/img_rgb.png', 'images/theme_rgb.png', 'images/theme_mask.png',
                                   'images/points_rgb.png', 'images/points_mask.png')
    test_ab_out = model.inference3_1(image_l_test, image_gra_test, theme_ab_test, theme_mask_test, point_ab_test, point_mask_test,
                                     is_training=False, scope_name='UserGuide')
    _, test_ab_out2 = model.gen_PRLNet(test_ab_out, image_l_test, 2, scope_name='PRLNet')
    test_rgb_out0 = \
        input_data.lab_to_rgb(tf.concat([(image_l_test + 1.) / 2 * 100., (test_ab_out + 1.) / 2 * 255. - 128], axis=3))
    test_rgb_out = \
        input_data.lab_to_rgb(tf.concat([(image_l_test + 1.) / 2 * 100., (test_ab_out2 + 1.) / 2 * 255. - 128], axis=3))
    test_psnr0 = 10 * tf.log(1 / (tf.reduce_mean(tf.square(test_rgb_out0 - image_rgb_test)))) / np.log(10)
    test_psnr = 10 * tf.log(1 / (tf.reduce_mean(tf.square(test_rgb_out - image_rgb_test)))) / np.log(10)

    var_list = tf.global_variables()
    var_model1 = [var for var in var_list if var.name.startswith('UserGuide')]
    var_model2 = [var for var in var_list if var.name.startswith('PRLNet')]
    var_total = var_model1 + var_model2
    paras_count1 = tf.reduce_sum([tf.reduce_prod(v.shape) for v in var_model1])
    paras_count2 = tf.reduce_sum([tf.reduce_prod(v.shape) for v in var_model2])
    print('UserGuide参数数目:%d' % sess.run(paras_count1))
    print('Detailed参数数目:%d' % sess.run(paras_count2))

    global_step = tf.train.get_or_create_global_step(sess.graph)
    # the loss function
    loss1, _ = model.loss_colorization(out_ab_batch2, image_ab_batch, index_ab_batch, l1, l2)
    loss2, _ = model.loss_residual(out_ab_batch, image_ab_batch)
    total_loss = loss1 + loss2
    train_op, learning_rate = model.training(total_loss, global_step, 1e-3, 4e4, 0.7, var_total)

    saver1 = tf.train.Saver(var_list=var_model1)
    saver2 = tf.train.Saver(var_list=var_model2)
    saver3 = tf.train.Saver(var_list=var_total)

    sess.run(tf.global_variables_initializer())  # 变量初始化
    print('载入检查点...')
    ckpt = tf.train.get_checkpoint_state('logs_1')
    if ckpt and ckpt.model_checkpoint_path:
        global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
        saver1.restore(sess, ckpt.model_checkpoint_path)
        print('载入成功, global_step = %s' % global_step)
    else:
        print('载入失败')

    ckpt = tf.train.get_checkpoint_state('logs_2')
    if ckpt and ckpt.model_checkpoint_path:
        global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
        saver2.restore(sess, ckpt.model_checkpoint_path)
        print('载入成功, global_step = %s' % global_step)
    else:
        print('载入失败')

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    s_t = time.time()
    try:
        for step in range(MAX_STEP):
            if coord.should_stop():
                print('???')
                break

            _, loss, lr = sess.run([train_op, total_loss, learning_rate])

            if step % 100 == 0:
                runtime = time.time() - s_t
                psnr0, psnr = sess.run([test_psnr0, test_psnr])
                print('Step: %d, Loss_total: %g, test_psnr0: %.2fdB, test_psnr: %.2fdB, learning_rate:%g, '
                      'time:%.2fs, time left: %.2fhours'
                      % (step, loss,  psnr0, psnr, lr, runtime, (MAX_STEP - step) * runtime / 360000))
                s_t = time.time()

            if step % 10000 == 0 or step == MAX_STEP - 1:
                if not os.path.exists(logs_dir):
                    os.makedirs(logs_dir)
                checkpoint_path = os.path.join(logs_dir, 'model.ckpt')
                saver3.save(sess, checkpoint_path, global_step=step)

            if step % 1000 == 0 or step == MAX_STEP - 1:
                test_out = sess.run(test_rgb_out)
                test_out = test_out[0]

                save_path = 'images/logs_output/' + logs_dir
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                save_path += 'step_' + str(step) + '.png'
                plt.imsave(save_path, test_out)

    except tf.errors.OutOfRangeError:
        print('Done.')
    finally:
        coord.request_stop()

    # 等待线程结束
    coord.join(threads=threads)
    sess.close()


if __name__ == '__main__':
    train3()
