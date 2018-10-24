import numpy as np
import tensorflow as tf
import utils
from os import path
import h5py


K_INIT = lambda: tf.truncated_normal_initializer(stddev=0.02)
B_INIT = lambda: tf.zeros_initializer()


class GS_Model:
    def __init__(self, batch_size=256, lr=1e-3, lr_decay=0.1, seed=309):
        tf.reset_default_graph()
        self.batch_size = batch_size
        self.lr = lr
        self.lr_decay = lr_decay
        self.data_dir = 'f:/dataset/AgriculturalDisease/'

        self.last_full_loss = 9.8
        self.seed = seed

        self.init_img_keys()

        self.build_placeholder()
        self.build_model()
        self.build_losses()
        self.build_train_ops()

        utils.show_params_scale()
        print('\nmodel_version: 1.0.5.1021')

    def build_placeholder(self):
        print('building placeholder')
        self.pl_in_data = tf.placeholder('float32', (None, 8, 8, 1920), 'in_data')
        self.pl_lr = tf.placeholder('float32', name='learning_rate')
        self.pl_training = tf.placeholder('bool', name='bn_training')
        self.pl_sr_lb = tf.placeholder('float32', [None, utils.sr_classes],
                                       'serious_classes')
        self.pl_sp_lb = tf.placeholder('float32', [None, utils.sp_classes],
                                       'spieces_classes')
        self.pl_ds_lb = tf.placeholder('float32', [None, utils.ds_classes],
                                       'disease_classes')
        self.pl_full_lb = tf.placeholder(
            'float32', [None, utils.full_classes], 'full_classes')


    def build_model(self):
        print('buliding model')
        with tf.variable_scope('sr_class', reuse=tf.AUTO_REUSE):
            self.build_sr_class_block()
        with tf.variable_scope('sp_class', reuse=tf.AUTO_REUSE):
            self.build_sp_class_block()
        with tf.variable_scope('ds_class', reuse=tf.AUTO_REUSE):
            self.build_ds_class_block()

        with tf.variable_scope('full_class', reuse=tf.AUTO_REUSE):
            self.build_full_class_block()
        # with tf.variable_scope('top_layer', reuse=tf.AUTO_REUSE):
        #     self.build_bypass_full_block()

        # self.out_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
        #         labels=self.pl_full_lb, logits=self.out_logits))
        # self.out_train_op = tf.train.AdamOptimizer(self.pl_lr).minimize(self.out_loss)


    def build_losses(self):
        print('building losses')
        with tf.variable_scope('losses', reuse=tf.AUTO_REUSE):
            self.sr_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
                labels=self.pl_sr_lb, logits=self.sr_logits))
            self.sp_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
                labels=self.pl_sp_lb, logits=self.sp_logits))
            self.ds_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
                labels=self.pl_ds_lb, logits=self.ds_logits))
            self.full_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
                labels=self.pl_full_lb, logits=self.full_logits))


    def build_train_ops(self):
        print('building optimizer & train_ops')
        optimizer = tf.train.AdamOptimizer(self.pl_lr)

        # 各自优化各自的网络
        all_tr_vars = tf.trainable_variables()
        sr_varlist, sp_varlist, ds_varlist, full_varlist = [], [], [], []
        for var in all_tr_vars:
            if var.name.startswith('sr_class'):
                sr_varlist.append(var)
            elif var.name.startswith('sp_class'):
                sp_varlist.append(var)
            elif var.name.startswith('ds_class'):
                ds_varlist.append(var)
            else:
                full_varlist.append(var)

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            self.sr_train_op = optimizer.minimize(self.sr_loss, var_list=sr_varlist)
            self.sp_train_op = optimizer.minimize(self.sp_loss, var_list=sp_varlist)
            self.ds_train_op = optimizer.minimize(self.ds_loss, var_list=ds_varlist)
            self.full_train_op = optimizer.minimize(self.full_loss, var_list=full_varlist)


    def build_sr_class_block(self):
        print('--- building sr cls block')
        sr_mid = tf.layers.conv2d(
            self.pl_in_data, 80, 1, 1, kernel_initializer=K_INIT(), use_bias=False)
        sr_mid = tf.layers.batch_normalization(sr_mid, training=self.pl_training)
        sr_mid = tf.nn.tanh(sr_mid)

        self.sr_mid = tf.reduce_max(sr_mid, axis=[1,2])
        sr_logits = tf.layers.batch_normalization(self.sr_mid, training=self.pl_training)

        sr_logits = tf.layers.flatten(sr_logits)
        self.sr_logits = tf.layers.dense(sr_logits, utils.sr_classes, kernel_initializer=K_INIT())


    def build_sp_class_block(self):
        print('--- building sp cls block')
        sp_mid = tf.layers.conv2d(
            self.pl_in_data, 24, 1, 1, kernel_initializer=K_INIT(), use_bias=False)
        sp_mid = tf.layers.batch_normalization(sp_mid, training=self.pl_training)
        sp_mid = tf.nn.tanh(sp_mid)

        self.sp_mid = tf.reduce_mean(sp_mid, axis=[1,2])
        sp_logits = tf.layers.batch_normalization(self.sp_mid, training=self.pl_training)

        sp_logits = tf.layers.flatten(sp_logits)
        self.sp_logits = tf.layers.dense(sp_logits, utils.sp_classes, kernel_initializer=K_INIT())


    def build_ds_class_block(self):
        print('--- building ds cls block')
        ds_mid = tf.layers.conv2d(
            self.pl_in_data, 80, 1, 1, kernel_initializer=K_INIT(), use_bias=False)
        ds_mid = tf.layers.batch_normalization(ds_mid, training=self.pl_training)
        ds_mid = tf.nn.tanh(ds_mid)

        self.ds_mid = tf.reduce_mean(ds_mid, axis=[1,2])
        ds_logits = tf.layers.batch_normalization(self.ds_mid, training=self.pl_training)

        ds_logits = tf.layers.flatten(ds_logits)
        self.ds_logits = tf.layers.dense(ds_logits, utils.ds_classes, kernel_initializer=K_INIT())

    def build_full_class_block(self):
        print('--- building full cls block')
        layerX_mid = tf.layers.conv2d(
            self.pl_in_data, 32, 1, 1, kernel_initializer=K_INIT(), use_bias=False)
        layerX_mid = tf.layers.batch_normalization(layerX_mid, training=self.pl_training)
        layerX_mid = tf.nn.tanh(layerX_mid)

        self.layerX_mid = tf.reduce_mean(layerX_mid, axis=[1,2])

        stack = tf.concat([
            self.layerX_mid, self.sr_mid, self.sp_mid, self.ds_mid,
            self.sr_logits, self.sp_logits, self.ds_logits
        ], axis=1)
        stack = tf.layers.batch_normalization(stack, training=self.pl_training)

        # layerX = tf.layers.flatten(layerX)
        # stack = tf.concat([self.sr_logits, self.sp_logits, self.ds_logits, layerX], axis=1)
        # flat = tf.layers.dropout(stack, training=self.pl_training)

        self.full_logits = tf.layers.dense(stack, utils.full_classes, kernel_initializer=K_INIT())


    def build_bypass_full_block(self):
        layerX = tf.reduce_mean(self.pl_in_data, axis=[1,2])
        layerX = tf.layers.dropout(layerX, training=self.pl_training)

        layerX = tf.layers.flatten(layerX)
        self.out_logits = tf.layers.dense(layerX, utils.full_classes)


    def init_img_keys(self):
        print('initiating image keys')
        np.random.seed(self.seed)

        self.tr_path = path.join(self.data_dir, 'all_tv-densenet.h5')

        self.tr_keys = fetch_keys(self.tr_path)
        self.tr_keys, self.v_keys, self.test_keys = self.tr_keys[:-8000], self.tr_keys[-6000:-3000], self.tr_keys[-3000:]
        
        np.random.shuffle(self.tr_keys)
        np.random.shuffle(self.v_keys)
        np.random.shuffle(self.test_keys)

        with h5py.File(self.tr_path, 'r') as f:
            k0, k1, k2 = [k[309] for k in (self.tr_keys, self.v_keys, self.test_keys)]
            print('key validation','-'*12)
            print('k0', hex(hash(f[k0]['img_data'].value.tostring() ) )[-8:])
            print('k1', hex(hash(f[k1]['img_data'].value.tostring() ) )[-8:])
            print('k2', hex(hash(f[k2]['img_data'].value.tostring() ) )[-8:])
            print('key validation','-'*12)

        print(f'tr: {len(self.tr_keys)}, v: {len(self.v_keys)} test: {len(self.test_keys)}')


    # def shuffle_keys(self):
    #     keys = self.tr_keys + self.test_keys
    #     np.random.shuffle(keys)
    #     self.tr_keys = keys[:-4000]
    #     self.test_keys = keys[-4000:]


    def get_feeder(self, key_type, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size

        data_path = self.tr_path
        if key_type == 'training':
            keys = self.tr_keys
        elif key_type == 'validation':
            keys = self.v_keys
        elif key_type == 'test':
            keys = self.test_keys
        else:
            raise ValueError(f'wrong key_type: {key_type}, must be "validation", "test"')

        with h5py.File(data_path, 'r') as fp:
            total = len(keys) // batch_size + 1
            np.random.shuffle(keys)
            for ii in range(total):
                batch_keys = keys[ii*batch_size : (ii+1)*batch_size]
                if len(batch_keys) == 0:
                    return
                print(f'step {ii+1}/{total} | ', end='\r')
                yield stack_data_from_h5(fp, batch_keys)

    
    def get_training_feeder(self):
        tr_file = 'F:/dataset/AgriculturalDisease/trainingset - augmeted.h5'
        batch_size = self.batch_size
        with h5py.File(tr_file, 'r') as fp:
            keys = [key for key in fp.keys()]
            np.random.shuffle(keys)
            total = len(keys) // batch_size + 1

            for ii in range(total):
                batch_keys = keys[ii*batch_size : (ii+1)*batch_size]
                if len(batch_keys) == 0:
                    return
                print(f'step {ii+1}/{total} | ', end='\r')
                yield self.stack_tr_data(fp, batch_keys)
    
    def stack_tr_data(self, fp, batch_keys):
        '''stack data for method: self.get_training_feeder'''
        # ['feat', 'img_id', 'lb']
        batch_raw_lb = []
        batch_feat = []
        for key in batch_keys:
            gp = fp[key]
            raw_lb = gp['lb'].value
            feat = gp['feat'].value
            batch_raw_lb.append(raw_lb)
            batch_feat.append(feat)
        
        sp, ds, sr, full_lb = utils.transfer_labels(np.array(batch_raw_lb))
        features = np.array(batch_feat)

        return features, sr, sp, ds, full_lb


    def train(self, epoch, ckpt_dir='../../ckpt/agr-ds', go_on=False):
        saver=tf.train.Saver()
        if not go_on:
            init_op = tf.global_variables_initializer()
            do_init = lambda sess: sess.run(init_op)
        else:
            model_file = tf.train.latest_checkpoint(ckpt_dir)
            do_init = lambda sess: saver.restore(sess, model_file)

        with tf.Session() as sess:
            do_init(sess)
            for ep in range(epoch):
                tr_feeder = self.get_feeder('training')
                for img_data, sr, sp, ds, full_lb in tr_feeder:
                    feed_dict = {
                        self.pl_lr: self.lr,
                        self.pl_training: True,
                        self.pl_in_data: img_data,
                        self.pl_sp_lb: sp,
                        self.pl_sr_lb: sr,
                        self.pl_ds_lb: ds,
                        self.pl_full_lb: full_lb,
                    }

                    sr_loss, sp_loss, ds_loss, full_loss, _,_,_,_ = sess.run(
                        [
                            self.sr_loss, self.sp_loss, self.ds_loss, self.full_loss,
                            self.sr_train_op, self.sp_train_op,
                            self.ds_train_op, self.full_train_op,
                        ], feed_dict=feed_dict)
                    L = [str(round(l, 6)) for l in (sr_loss, sp_loss, ds_loss, full_loss)]
                    print(f'--------losses: {L}', end='\r')

                    # break; # for debug

                v_feeder = self.get_feeder('validation', batch_size=1024)
                self.do_validation_saving(sess, v_feeder, saver, ckpt_dir, ep)
                if (ep+1)%5 == 0:
                    self.lr *= self.lr_decay


    def do_validation_saving(self, sess, v_feeder, saver, ckpt_dir, ep=0):
        losses = []
        for img_data, sr, sp, ds, full_lb in v_feeder:
            feed_dict = {
                self.pl_lr: self.lr,
                self.pl_training: False,
                self.pl_in_data: img_data,
                self.pl_sr_lb: sr,
                self.pl_sp_lb: sp,
                self.pl_ds_lb: ds,
                self.pl_full_lb: full_lb,
            }
            L = sess.run([
                self.sr_loss, self.sp_loss, self.ds_loss, self.full_loss,
                # self.out_loss
            ], feed_dict=feed_dict)
            losses.append(L)
        L = np.mean(losses, axis=0)

        # if this full_loss is smaller then last, do saving
        try:
            if L[3] < self.last_full_loss:
                print(f'current full_loss: {L[3]}, last: {self.last_full_loss}, saving model')
                saver.save(sess, path.join(ckpt_dir,'model.ckpt'), global_step=ep)
                self.last_full_loss = L[3]

            # beautify number display
            L = [str(round(l, 6)) for l in L]
            # print(f'losses: sr -> {L[0]}, sp -> {L[1]}, ds -> {L[2]}, full -> {L[3]}')
            print(f'valid loss: {L}')

        except:
            print('cause some err')

    def pred_on_batch(self, batch_keys, h5fp, sess):
        pred_op = tf.nn.softmax(self.full_logits)
        stack_features = []
        for key in batch_keys:
            features = h5fp[key]
            stack_features.append(features)
        stack_features = np.array(stack_features)
        pred = sess.run(pred_op,feed_dict={
            self.pl_in_data: stack_features,
            self.pl_training: False,
        })
        pred = np.argmax(pred, axis=1)
        json_data = []
        for idx, key in enumerate(batch_keys):
            p = {}
            p['image_id'] = key
            p['disease_class'] = pred[idx]
            json_data.append(p)
        return json_data

    def pred_test(self):
        feat_path = 'f:/dataset/AgriculturalDisease/test.h5'
        json_data = []

        # load session
        sess = tf.Session()
        saver = tf.train.Saver()
        model_path = tf.train.latest_checkpoint('../../ckpt/agr-ds/')
        saver.restore(sess, model_path)

        with h5py.File(feat_path, 'r') as f:
            keys = [key for key in f.keys()]
            total = len(keys)//1024 + 1
            print('total: ', total)

            for i in range(total):
                print(f'dealing with batch {i+1}/{total}')
                batch_keys = keys[i*1024 : (i+1)*1024]
                pred_json = self.pred_on_batch(batch_keys, f, sess)
                json_data += pred_json

        sess.close()
        return json_data

# 18 24 80 32
##--------------------------------------------------

def fetch_keys(data_path):
    with h5py.File(data_path, 'r') as f:
        keys = list(f.keys())
        np.random.shuffle(keys)
        return keys


def stack_data_from_h5(h5fp, batch_keys):
    img_data, sr, sp, ds, full_lb = [], [], [], [], []
    for key in batch_keys:
        record = h5fp[key]
        img_data.append(record['img_data'].value)
        sr.append(record['sr'].value)
        sp.append(record['sp'].value)
        ds.append(record['ds'].value)
        full_lb.append(record['full_lb'].value)

    return [np.array(d) for d in (img_data, sr, sp, ds, full_lb)]