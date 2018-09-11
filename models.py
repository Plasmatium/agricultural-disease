import numpy as np
import tensorflow as tf
import utils

from layers import brc, dense_block
from data_flow import DataFlow

class Model:

    def __init__(self, batch_size=256):
        tf.reset_default_graph()

        self.dataflow = DataFlow(batch_size)
        self.pl_inData = tf.placeholder(tf.float32, [None, 256, 256, 3], 'in_data')
        self.pl_training = tf.placeholder(tf.bool, name='bn_training')
        self.pl_lr = tf.placeholder(tf.float32, name='learning_rate')
        
        self.pl_sr_lb = tf.placeholder(tf.float32, [None, utils.sr_classes], 'serious_classes')
        self.pl_sp_lb = tf.placeholder(tf.float32, [None, utils.sp_classes], 'spieces_classes')
        self.pl_ds_lb = tf.placeholder(tf.float32, [None, utils.ds_classes], 'disease_classes')
        self.pl_full_lb = tf.placeholder(tf.float32, [None, utils.full_classes], 'full_classes')

        self.build_model()
        self.build_loss()
        self.build_train_op()

    def build_model(self):
        K_INIT = lambda: tf.truncated_normal_initializer(stddev=0.02)
        B_INIT = lambda: tf.zeros_initializer()

        d = tf.layers.conv2d(self.pl_inData, 32, 7, 2, kernel_initializer=K_INIT(), bias_initializer=B_INIT())

        d = dense_block('dense_block_1', d, training=self.pl_training)
        d = tf.layers.max_pooling2d(d, 3, 2, 'same')

        d = dense_block('dense_block_2', d, filters=24, training=self.pl_training)
        d = tf.layers.average_pooling2d(d, 3, 2, 'same')

        # sr classes
        preout1 = tf.reduce_mean(d, [1, 2])
        preout1 = tf.layers.dropout(preout1)
        self.out1_logits = tf.layers.dense(preout1, utils.sr_classes, kernel_initializer=K_INIT(), bias_initializer=B_INIT())
        # print(out1_logits)

        d = dense_block('dense_block_3', d, filters=28, training=self.pl_training)
        d = tf.layers.average_pooling2d(d, 3, 2, 'same')

        d = dense_block('dense_block_4', d, filters=32, training=self.pl_training)
        d = tf.layers.average_pooling2d(d, 3, 2, 'same')

        # sp class
        preout2 = tf.reduce_mean(d, [1, 2])
        preout2 = tf.layers.dropout(preout2)
        self.out2_logits = tf.layers.dense(preout2, utils.sp_classes, kernel_initializer=K_INIT(), bias_initializer=B_INIT())
        # print(out2_logits)

        d = dense_block('dense_block_5', d, filters=36, training=self.pl_training)
        d = tf.layers.average_pooling2d(d, 3, 2, 'same')

        d = dense_block('dense_block_6', d, filters=40, training=self.pl_training)
        d = tf.layers.average_pooling2d(d, 3, 2, 'same')

        # ds class
        preout3 = tf.reduce_mean(d, [1, 2])
        preout3 = tf.layers.dropout(preout3)
        self.out3_logits = tf.layers.dense(preout3, utils.ds_classes, kernel_initializer=K_INIT(), bias_initializer=B_INIT())
        # print(out3_logits)

        d = dense_block('dense_block_7', d, filters=44, training=self.pl_training)
        d = tf.layers.average_pooling2d(d, 3, 2, 'same')

        d = dense_block('dense_block_8', d, filters=48, training=self.pl_training)
        d = tf.layers.average_pooling2d(d, 3, 2, 'same')

        # summary outputs
        preout4 = tf.reduce_mean(d, [1, 2])
        preout4 = tf.layers.dropout(preout4)
        self.out4_logits = tf.layers.dense(preout4, utils.full_classes, kernel_initializer=K_INIT(), bias_initializer=B_INIT())
        # print(out4_logits)
        
    def build_loss(self):
        self.sr_loss = tf.nn.softmax_cross_entropy_with_logits_v2(
            labels=self.pl_sr_lb, logits=self.out1_logits)
        self.sp_loss = tf.nn.softmax_cross_entropy_with_logits_v2(
            labels=self.pl_sp_lb, logits=self.out2_logits)
        self.ds_loss = tf.nn.softmax_cross_entropy_with_logits_v2(
            labels=self.pl_ds_lb, logits=self.out3_logits)
        self.full_loss = tf.nn.softmax_cross_entropy_with_logits_v2(
            labels=self.pl_full_lb, logits=self.out4_logits)
    
    def build_train_op(self):
        # 4个优化器
        with tf.variable_scope('opt', reuse=tf.AUTO_REUSE):
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                for opt_name in ('sr', 'sp', 'ds', 'full'):
                    setattr(self, 'opt_'+opt_name, tf.train.AdamOptimizer(self.pl_lr))
                    opt = getattr(self, 'opt_'+opt_name)
                    loss = getattr(self, opt_name+'_loss')
                    setattr(self, 'train_op_'+opt_name, opt.minimize(loss))

    def train(self, epoches=10):
        init_op = tf.global_variables_initializer()
        with tf.Session() as sess:
            sess.run(init_op)
            for ep in range(epoches):
                feeder = self.dataflow.feed('training')

                for idx, (img_data, labels) in enumerate(feeder):
                    print(idx, end='\r')
                    sp, ds, sr, full_lb = labels
                    feed_dict = {
                        self.pl_inData: img_data,
                        self.pl_sp_lb: sp,
                        self.pl_ds_lb: ds,
                        self.pl_sr_lb: sr,
                        self.pl_full_lb: full_lb,
                        self.pl_training: True,
                        self.pl_lr: 1e-4
                    }
                    sess.run(
                        [self.train_op_sr, self.train_op_sp, self.train_op_ds, self.train_op_full],
                        feed_dict=feed_dict)
                

