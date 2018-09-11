import numpy as np
import tensorflow as tf

# 构建稠密块

def brc(name, data, filters, training, w_s=[3,1]):
    '''bn - relu - conv'''
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        data = tf.layers.batch_normalization(data, training=training)
        data = tf.nn.relu(data)
        w, s = w_s
        data = tf.layers.conv2d(data, filters, w, s, padding='same', use_bias=False,
            kernel_initializer=tf.truncated_normal_initializer(stddev=0.02))
        return data

def dense_block(name, in_data, filters=20, depth=5, k=16, training=True):
    cc = [in_data]
    dense = in_data

    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        for i in range(depth):
            dense = brc(f'brc1x1_{i}', dense, k, training, [1,1])
            dense = brc(f'brc3x1_{i}', dense, filters, training, [3,1])
            cc.append(dense)
            dense = tf.concat(cc, axis=3)            
        return dense
