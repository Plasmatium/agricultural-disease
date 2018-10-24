import numpy as np
import tensorflow as tf

# depr
def instance_norm(x, name='instance_norm'):
    ''' copy from https://github.com/hardikbansal/CycleGAN/blob/master/layers.py '''
    with tf.variable_scope(name, reuse=False):
        epsilon = 1e-5
        mean, var = tf.nn.moments(x, [1, 2], keep_dims=True)
        scale = tf.get_variable(
            'scale', [x.get_shape()[-1]],
            initializer=tf.truncated_normal_initializer(mean=1.0, stddev=0.02))
        offset = tf.get_variable(
            'offset', [x.get_shape()[-1]],
            initializer=tf.constant_initializer(0.0))
        out = scale * tf.div(x - mean, tf.sqrt(var + epsilon)) + offset

    return out


def brc(name, data, filters, training, w_s=[3,1]):
    '''bn - relu - conv'''
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        # data = tf.layers.batch_normalization(data, training=training)
        data = tf.contrib.layers.instance_norm(data)
        data = tf.nn.relu(data)
        w, s = w_s
        data = tf.layers.conv2d(data, filters, w, s, padding='same', use_bias=False,
            kernel_initializer=tf.truncated_normal_initializer(stddev=0.02))
        return data


def dense_block(name, in_data, filters=8, depth=5, k=12, training=True):
    cc = [in_data]
    dense = in_data

    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        for i in range(depth):
            dense = brc(f'brc1x1_{i}', dense, k, training, [1,1])
            dense = brc(f'brc3x1_{i}', dense, filters, training, [3,1])
            cc.append(dense)
            dense = tf.concat(cc, axis=3)
        return dense
