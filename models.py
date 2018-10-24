import numpy as np
import tensorflow as tf
import utils

from layers import brc, dense_block, instance_norm
from data_flow import DataFlow


class Model:
    def __init__(self, batch_size=256):
        tf.reset_default_graph()
        self.batch_size = batch_size

        self.dataflow = DataFlow(batch_size)
        self.pl_inData = tf.placeholder(tf.float32, [None, 256, 256, 3],
                                        'in_data')
        self.pl_training = tf.placeholder(tf.bool, name='bn_training')
        self.pl_lr = tf.placeholder(tf.float32, name='learning_rate')

        self.pl_sr_lb = tf.placeholder(tf.float32, [None, utils.sr_classes],
                                       'serious_classes')
        self.pl_sp_lb = tf.placeholder(tf.float32, [None, utils.sp_classes],
                                       'spieces_classes')
        self.pl_ds_lb = tf.placeholder(tf.float32, [None, utils.ds_classes],
                                       'disease_classes')
        self.pl_full_lb = tf.placeholder(
            tf.float32, [None, utils.full_classes], 'full_classes')

        self.build_model()
        self.build_loss()
        self.build_train_op()

        self.saver = tf.train.Saver()

    def build_model(self):
        K_INIT = lambda: tf.truncated_normal_initializer(stddev=0.02)
        B_INIT = lambda: tf.zeros_initializer()

        d = tf.layers.conv2d(
            self.pl_inData, 8, 7, 2, kernel_initializer=K_INIT(), use_bias=False)

        d = dense_block(
            'dense_block_1', d, filters=12, training=self.pl_training)
        d = tf.layers.max_pooling2d(d, 3, 2, 'same')

        d = dense_block(
            'dense_block_2', d, filters=16, training=self.pl_training)
        d = tf.layers.average_pooling2d(d, 3, 2, 'same')
        print('o1', d)
        # sr classes
        preout1 = tf.reduce_mean(d, [1, 2])
        preout1 = tf.layers.dropout(preout1, training=self.pl_training)
        self.out1_logits = tf.layers.dense(
            preout1,
            utils.sr_classes,
            kernel_initializer=K_INIT(),
            bias_initializer=B_INIT())
        # print(out1_logits)

        d = dense_block(
            'dense_block_3', d, filters=20, training=self.pl_training)
        d = tf.layers.average_pooling2d(d, 3, 2, 'same')

        d = dense_block(
            'dense_block_4', d, filters=24, training=self.pl_training)
        d = tf.layers.average_pooling2d(d, 3, 2, 'same')
        print('o2', d)
        # sp class
        preout2 = tf.reduce_mean(d, [1, 2])
        preout2 = tf.layers.dropout(preout2, training=self.pl_training)
        self.out2_logits = tf.layers.dense(
            preout2,
            utils.sp_classes,
            kernel_initializer=K_INIT(),
            bias_initializer=B_INIT())
        # print(out2_logits)

        d = dense_block(
            'dense_block_5', d, filters=28, training=self.pl_training)
        d = tf.layers.average_pooling2d(d, 3, 2, 'same')

        d = dense_block(
            'dense_block_6', d, filters=32, training=self.pl_training)
        d = tf.layers.average_pooling2d(d, 3, 2, 'same')
        print('o3', d)
        # ds class
        preout3 = tf.reduce_mean(d, [1, 2])
        preout3 = tf.layers.dropout(preout3, training=self.pl_training)
        self.out3_logits = tf.layers.dense(
            preout3,
            utils.ds_classes,
            kernel_initializer=K_INIT(),
            bias_initializer=B_INIT())
        # print(out3_logits)
        with tf.variable_scope('top_layers', reuse=tf.AUTO_REUSE):
            d = dense_block(
                'dense_block_7', d, filters=36, training=self.pl_training)
            d = tf.layers.average_pooling2d(d, 3, 2, 'same')

            # d = dense_block(
            #     'dense_block_8', d, filters=26, training=self.pl_training)
            # d = tf.layers.average_pooling2d(d, 3, 2, 'same')
            print('o4', d)
            # summary outputs
            preout4 = tf.reduce_mean(d, [1, 2])
            preout4 = tf.layers.dropout(preout4, training=self.pl_training)
            fc_layer = tf.layers.dense(
                preout4,
                32,
                kernel_initializer=K_INIT(),
                bias_initializer=B_INIT())
            fc_layer = tf.concat(
                [fc_layer, self.out1_logits, self.out2_logits, self.out3_logits],
                axis=1)
            self.out4_logits = tf.layers.dense(
                fc_layer,
                utils.full_classes,
                kernel_initializer=K_INIT(),
                bias_initializer=B_INIT())
            # print(out4_logits)

    def build_loss(self):
        with tf.variable_scope('losses', reuse=tf.AUTO_REUSE):
            self.sr_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
                labels=self.pl_sr_lb, logits=self.out1_logits))
            self.sp_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
                labels=self.pl_sp_lb, logits=self.out2_logits))
            self.ds_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
                labels=self.pl_ds_lb, logits=self.out3_logits))
            self.full_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
                labels=self.pl_full_lb, logits=self.out4_logits))

    def build_train_op(self):
        # 4个优化器
        with tf.variable_scope('opt', reuse=tf.AUTO_REUSE):
            # update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            # with tf.control_dependencies(update_ops):
            for opt_name in ('sr', 'sp', 'ds'):
                setattr(self, 'opt_' + opt_name,
                        tf.train.AdamOptimizer(self.pl_lr))
                opt = getattr(self, 'opt_' + opt_name)
                loss = getattr(self, opt_name + '_loss')
                setattr(self, 'train_op_' + opt_name, opt.minimize(loss))
            
            # 顶层需要控制只更新后面层
            self.opt_full = tf.train.AdamOptimizer(self.pl_lr)
            top_layers_var = [v for v in tf.trainable_variables() if v.name.startswith('top_layers')]
            self.train_op_full = self.opt_full.minimize(self.full_loss, var_list=top_layers_var)

    def build_validation(self):
        self.v_feeder = self.dataflow.feed('validation')

    def train(self, epoches=10, learning_rate=4e-4):
        init_op = tf.global_variables_initializer()
        with tf.Session() as sess:
            sess.run(init_op)
            for ep in range(epoches):
                feeder = self.dataflow.feed('training')

                for idx, (img_data, labels) in enumerate(feeder):
                    sp, ds, sr, full_lb = labels
                    for i in range(self.batch_size//8):
                        inData = img_data[i*8:(i+1)*8]
                        lbData = labels[i*8:(i+1)*8]
                        sp_ = sp[i*8:(i+1)*8]
                        ds_ = ds[i*8:(i+1)*8]
                        sr_ = sr[i*8:(i+1)*8]
                        full_lb_ = full_lb[i*8:(i+1)*8]

                        feed_dict = {
                            self.pl_inData: inData,
                            self.pl_sp_lb: sp_,
                            self.pl_ds_lb: ds_,
                            self.pl_sr_lb: sr_,
                            self.pl_full_lb: full_lb_,
                            self.pl_training: True,
                            self.pl_lr: learning_rate
                        }
                        sr_loss, sp_loss, ds_loss, full_loss, _, _, _, _ = sess.run(
                            [
                                self.sr_loss, self.sp_loss, self.ds_loss,
                                self.full_loss, self.train_op_sr, self.train_op_sp,
                                self.train_op_ds, self.train_op_full
                            ],
                            feed_dict=feed_dict)

                        info = f'{idx}, losses: sr -> {sr_loss}, sp -> {sp_loss}, ds -> {ds_loss}, full_loss -> {full_loss}'
                        print(info, end='\r')
                    
                    self.saver.save(sess, '../../ckpt/agr-ds/model.ckpt', global_step=ep*10000+idx)
                learning_rate /= 2.0