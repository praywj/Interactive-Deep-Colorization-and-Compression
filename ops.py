import tensorflow as tf
import tensorflow.contrib.layers as layers


# group normalization
def group_norm(inputs, G=32, eps=1e-5, name='GroupNorm'):
    with tf.variable_scope(name):
        N, H, W, C = inputs.shape
        gamma = tf.get_variable(name='gamma',
                                shape=[1, 1, 1, C],
                                dtype=tf.float32,
                                initializer=tf.constant_initializer(1))
        beta = tf.get_variable(name='beta',
                               shape=[1, 1, 1, C],
                               dtype=tf.float32,
                               initializer=tf.zeros_initializer())

        inputs = tf.reshape(inputs, [N, G, H, W, C // G])
        mean, var = tf.nn.moments(inputs, [2, 3, 4], keep_dims=True)
        inputs = (inputs - mean) / tf.sqrt(var + eps)
        x = tf.reshape(inputs, [N, H, W, C])

    return x * gamma + beta


# convolution layer
def conv2d(inputs, output_dim, kernel_size, stride, dilation=1, padding='SAME',
           activation_fn=None, norm_fn=None, is_training=True, scope_name=None):
    """
    Convolution for 2D
    :param inputs: A 4-D tensor
    :param output_dim: A int
    :param kernel_size: A int
    :param stride: A int
    :param dilation: A int
    :param padding: 'SAME' or 'VALID'
    :param activation_fn: A function handle
    :param norm_fn: A function handle
    :param is_training: True or False
    :param scope_name: A string
    :return: A 4-D tensor
    """
    with tf.variable_scope(scope_name):
        conv = tf.layers.conv2d(inputs=inputs,
                                filters=output_dim,
                                kernel_size=kernel_size,
                                strides=stride,
                                dilation_rate=dilation,
                                padding=padding,
                                use_bias=False,
                                bias_initializer=layers.xavier_initializer())

        # normalization function
        if norm_fn is None:
            biases = tf.get_variable(name='b',
                                     shape=[output_dim],
                                     dtype=tf.float32,
                                     initializer=tf.zeros_initializer())
            conv = conv + biases
        elif norm_fn is tf.contrib.layers.batch_norm:
            conv = norm_fn(inputs=conv,
                           updates_collections=None,
                           is_training=is_training)
        elif norm_fn is tf.layers.batch_normalization:
            conv = norm_fn(inputs=conv,
                           axis=3,
                           epsilon=1e-5,
                           momentum=0.9,
                           training=is_training,
                           gamma_initializer=tf.random_uniform_initializer(1.0, 0.02))
        elif norm_fn is group_norm:  # 何凯明《Group Normalization》
            conv = norm_fn(inputs=conv,
                           G=32,
                           eps=1e-5)
        else:
            raise NameError

        # activation function
        if activation_fn is None:
            return conv
        else:
            return activation_fn(conv)


# depth-wise convolution layer(only for user-guided)
def depth_wise_conv2d(inputs, multiplier, kernel_size, stride, padding='SAME',
                      activation_fn=None, scope_name=None):
    """
    Depth-wise convolution for 2D
    :param inputs:  A 4-D tensor
    :param multiplier:  A int
    :param kernel_size: A int
    :param stride:  A int
    :param padding: 'SAME' or 'VALID'
    :param activation_fn:   A function handle
    :param scope_name:  A string
    :return:    A 4-D tensor
    """
    with tf.variable_scope(scope_name):
        # convolution
        weights = tf.constant(name='w',
                              value=1.,
                              shape=[kernel_size, kernel_size, inputs.get_shape()[-1], multiplier],
                              dtype=tf.float32)
        conv = tf.nn.depthwise_conv2d(input=inputs,
                                      filter=weights,
                                      strides=[1, stride, stride, 1],
                                      padding=padding)

        # activation function
        if activation_fn is None:
            return conv
        else:
            return activation_fn(conv)


# convolution transpose layer
def conv2d_transpose(inputs, output_dim, kernel_size, stride, padding='SAME',
                     activation_fn=None, norm_fn=None, is_training=True, scope_name=None):
    """
    Deconvolution for 2D
    :param inputs: A 4-D tensor
    :param output_dim: A int
    :param kernel_size: A int
    :param stride: A int
    :param padding: 'SAME' or 'VALID'
    :param activation_fn: A function handle
    :param norm_fn: A function handle
    :param is_training: True or False
    :param scope_name: A string
    :return: A 4-D tensor
    """
    with tf.variable_scope(scope_name):
        # deconvolution
        weights = tf.get_variable(name='w',
                                  shape=[kernel_size, kernel_size, output_dim, inputs.get_shape()[-1].value],
                                  dtype=tf.float32,
                                  initializer=layers.xavier_initializer())
        # initializer=tf.random_normal_initializer(0, 0.02))
        output_shape = inputs.get_shape().as_list()
        output_shape[1] *= 2
        output_shape[2] *= 2
        output_shape[3] = output_dim
        deconv = tf.nn.conv2d_transpose(value=inputs,
                                        filter=weights,
                                        output_shape=output_shape,
                                        strides=[1, stride, stride, 1],
                                        padding=padding)

        # normalization function
        if norm_fn is None:
            biases = tf.get_variable(name='b',
                                     shape=[output_dim],
                                     dtype=tf.float32,
                                     initializer=tf.zeros_initializer())
            deconv = deconv + biases
        elif norm_fn is tf.contrib.layers.batch_norm:
            deconv = norm_fn(inputs=deconv,
                             updates_collections=None,
                             is_training=is_training)
        elif norm_fn is tf.layers.batch_normalization:
            deconv = norm_fn(inputs=deconv,
                             axis=3,
                             epsilon=1e-5,
                             momentum=0.9,
                             training=is_training,
                             gamma_initializer=tf.random_uniform_initializer(1.0, 0.02))
        elif norm_fn is group_norm:  # 何凯明《Group Normalization》
            deconv = norm_fn(inputs=deconv,
                             G=32,
                             eps=1e-5)
        else:
            raise NameError

        # activation function
        if activation_fn is None:
            return deconv
        else:
            return activation_fn(deconv)