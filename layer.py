import numpy as np
import tensorflow as tf

# 构建稠密块

def brc(name, data, filters, training):
    '''bn - relu - conv'''
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        data = tf.layers.batch_normalization(data, training=training)
        data = tf.nn.relu(data)
        data = tf.layers.conv2d(data, filters, 3, 1, padding='same', use_bias=False,
            kernel_initializer=tf.truncated_normal_initializer(stddev=0.02))
        return data

def dense_block(name, in_data, filters, depth=5, training=True):
    cc = [in_data]
    dense = in_data

    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        for i in range(depth):
            dense = brc(f'dncv_{i}', dense, filters, training)
            cc.append(dense)
            dense = tf.concat(cc, axis=3)            
        return dense
