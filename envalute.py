import model
import input_data
import matplotlib.pyplot as plt
import tensorflow as tf
from skimage import measure
import os


def envalute():
    CAPACITY = 1000
    IMAGE_SIZE = 256
    BATCH_SIZE = 1
    save_path = 'test'
    # the number og local inputs u want to set
    num = [4]
    for x in num:
        image_color_dir = 'K:\\UserGuide_256\\test\\color'
        color_map_dir = 'K:\\UserGuide_256\\test\\color_map'
        theme_dir = 'K:\\UserGuide_256\\test\\color_theme'
        theme_mask_dir = 'K:\\UserGuide_256\\test\\color_theme_mask'
        local_dir = 'K:\\test_for_colorization\\Sample_mask\\slic_' + str(x)
        local_mask_dir = 'K:\\test_for_colorization\\Sample_mask\\slic_' + str(x)

        logs_dir = 'logs_3'
        sess = tf.Session()
        train_list = input_data.get_train_list(
            [image_color_dir, theme_dir, theme_mask_dir, color_map_dir, local_dir, local_mask_dir],
            ['color img', 'theme img', 'theme mask', 'color_map img', 'local img', 'local mask'],
            ['*', '*', 'png', 'png', 'png', 'png'], shuffle=False)

        image_rgb_batch, theme_rgb_batch, theme_mask_batch, index_rgb_batch, point_rgb_batch, point_mask_batch = \
            input_data.get_batch(train_list, IMAGE_SIZE, BATCH_SIZE, CAPACITY, False)

        image_lab_batch = input_data.rgb_to_lab(image_rgb_batch)
        image_l_batch = tf.reshape(image_lab_batch[:, :, :, 0] / 100.0 * 2 - 1, [BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, 1])
        image_l_gra_batch = model.sobel(image_l_batch)
        image_ab_batch = (image_lab_batch[:, :, :, 1:] + 128.) / 255.0 * 2 - 1

        theme_lab_batch = input_data.rgb_to_lab(theme_rgb_batch)
        theme_ab_batch = (theme_lab_batch[:, :, :, 1:] + 128.) / 255.0 * 2 - 1

        point_lab_batch = input_data.rgb_to_lab(point_rgb_batch)
        point_ab_batch = (point_lab_batch[:, :, :, 1:] + 128.) / 255.0 * 2 - 1

        # TODO: colorization
        out_ab_batch = model.inference3_1(image_l_batch, image_l_gra_batch, theme_ab_batch, theme_mask_batch,
                                          point_ab_batch, point_mask_batch,
                                          is_training=False, scope_name='UserGuide')
        # TODO: residual network
        _, out_ab_batch2 = model.gen_PRLNet(out_ab_batch, image_l_batch, 2, scope_name='PRLNet')
        test_rgb_out2 = \
            input_data.lab_to_rgb(
                tf.concat([(image_l_batch + 1.) / 2 * 100., (out_ab_batch2 + 1.) / 2 * 255. - 128], axis=3))

        var_list = tf.global_variables()
        var_model1 = [var for var in var_list if var.name.startswith('UserGuide')]
        var_model2 = [var for var in var_list if var.name.startswith('PRLNet')]
        var_total = var_model1 + var_model2
        paras_count1 = tf.reduce_sum([tf.reduce_prod(v.shape) for v in var_model1])
        paras_count2 = tf.reduce_sum([tf.reduce_prod(v.shape) for v in var_model2])
        print('UserGuide参数数目:%d' % sess.run(paras_count1))
        print('Detailed参数数目:%d' % sess.run(paras_count2))

        saver1 = tf.train.Saver(var_list=var_total)
        print('载入检查点...')
        ckpt = tf.train.get_checkpoint_state(logs_dir)
        if ckpt and ckpt.model_checkpoint_path:
            global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
            saver1.restore(sess, ckpt.model_checkpoint_path)
            print('载入成功, global_step = %s' % global_step)
        else:
            print('载入失败')

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        # compute the average psnr
        avg_psnr = 0.
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        try:
            for t in range(len(train_list[0])):
                in_rgb, out_rgb2 = sess.run([image_rgb_batch, test_rgb_out2])
                in_rgb = in_rgb[0]
                out_rgb2 = out_rgb2[0]
                psnr = measure.compare_psnr(out_rgb2, in_rgb)
                avg_psnr += psnr
                plt.imsave(save_path + '\\' + train_list[0][t].split('\\')[-1], out_rgb2)
                print('%s\n' % str(psnr))
            print('avg_psnr = %s' % str(avg_psnr / len(train_list[0])))

        except tf.errors.OutOfRangeError:
            print('Done.')
        finally:
            coord.request_stop()

        # 等待线程结束
        coord.join(threads=threads)
        sess.close()


if __name__ == '__main__':
    envalute()
