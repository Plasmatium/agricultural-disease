import numpy as np
import tensorflow as tf
import utils
import json
from glob import glob
from PIL import Image
from os import path
import h5py
import time
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import gc

base_dir = path.join('f:', 'dataset/AgriculturalDisease')
tr_dir = path.join(base_dir, 'trainingset')
v_dir = path.join(base_dir, 'validationset')
t_dir = path.join(base_dir, 'testA')


class DataFlow:

    def __init__(self, batch_size=256):
        self.batch_size = batch_size
        self.tr_dir = tr_dir
        self.v_dir = v_dir
        self.t_dir = t_dir
        self.reset()

    def init_dataIdx(self):
        file = path.join(self.v_dir, 'labels.json')
        with open(file) as f:
            v_idx = json.load(f)
            v_idx = [v for v in v_idx if '副本' not in v['image_id']]

        file = path.join(self.tr_dir, 'labels.json')
        with open(file) as f:
            tr_idx = json.load(f)
            tr_idx = [t for t in tr_idx if '副本' not in t['image_id']]

        itsc = set([t['image_id'] for t in tr_idx]).intersection(set([v['image_id'] for v in v_idx]))
        tr_idx = [t for t in tr_idx if t['image_id'] not in itsc]

        self.tr_dataIdx = tr_idx
        self.v_dataIdx = v_idx        

    def reset(self):
        self.init_dataIdx()
        self.currIdx = 0
        np.random.shuffle(self.tr_dataIdx)
        np.random.shuffle(self.v_dataIdx)

    def feed(self, datatype='training'):
        if datatype == 'training':
            dataIdx = self.tr_dataIdx
            img_dir = path.join(base_dir, 'trainingset/images')
        elif datatype == 'validation':
            dataIdx = self.v_dataIdx
            img_dir = path.join(base_dir, 'validationset/images')
        else:
            raise ValueError('datatype param in feed must be "training" or "validation"')

        self.total = len(dataIdx) // self.batch_size
        print(f'this epoch has total {self.total} batches')
        for i in range(self.total+1):
            meta = dataIdx[i*self.batch_size : (i+1)*self.batch_size]
            if meta == []:
                return

            img_indeces = [m['image_id'] for m in meta]
            img_files = img_indeces
            raw_labels = np.array([m['disease_class'] for m in meta])

            img_paths = [path.join(img_dir, fn) for fn in img_files]
            img_data = _extract_imgs(img_paths)
            sp, ds, sr, full_lb = utils.transfer_labels(raw_labels)
            yield img_data, sp, ds, sr, full_lb, img_indeces

# 数据densetnet转换
def _gs_transfer(dataset, h5fp, gsm):
    img_data, sp, ds, sr, full_lb, img_indeces = dataset
    features = gsm.predict(img_data)
    for idx, img_id in enumerate(img_indeces):
        try:
            group = h5fp.create_group(img_id)
        except:
            h5fp.pop(img_id)
            group = h5fp.create_group(img_id)

        group['img_data'] = features[idx]
        group.create_dataset('sp', data=sp[idx], compression='lzf')
        group.create_dataset('ds', data=ds[idx], compression='lzf')
        group.create_dataset('sr', data=sr[idx], compression='lzf')
        group.create_dataset('full_lb', data=full_lb[idx], compression='lzf')

def gs_transfer(gsm, suf, store_dir='f:/dataset/AgriculturalDisease/'):
    # gsm = tf.keras.applications.DenseNet121(
    #     include_top=False, input_shape=[256, 256, 3])
    # gsm = tf.keras.applications.DenseNet121([256, 256, 3], include_top=False)
    # gsm = tf.keras.applications.DenseNet201(input_shape=[256, 256, 3], include_top=False)
    # print('gsm loaded')
    df = DataFlow(1024)
    print('dataflow loaded')
    tr_feeder = df.feed('training')
    v_feeder = df.feed('validation')

    tr_path = path.join(store_dir, suf+'-training.h5')
    v_path = path.join(store_dir, suf+'-validation.h5')

    print('begin transfer')
    # with h5py.File(all_path) as fp:
    with h5py.File(tr_path, 'w') as fp:
        for idx, dataset in enumerate(tr_feeder):
            print(f'tr progress: {idx+1}', end='\r')
            _gs_transfer(dataset, fp, gsm)
        print('tr transfer finished')

    with h5py.File(v_path, 'w') as fp:
        for idx, dataset in enumerate(v_feeder):
            print(f'v progress: {idx+1}', end='\r')
            _gs_transfer(dataset, fp, gsm)
        print('v transfer finished')


def _extract_imgs(img_paths):
    rslt = [_refine(img_path) for img_path in img_paths]
    return np.array(rslt)


def _refine(img_path):
    ''' 经过检查，所有图片均长+宽均>=512，至少有一边是大于256的，对此进行缩放并做pad，
      img_path是图片路径，函数最终输出numpy数组 '''
    img_data = Image.open(img_path)
    # resize到长边256
    w, h = img_data.size
    if w > h:
        nw, nh = 256, int(h * 256 / w)
    else:
        nw, nh = int(w * 256 / h), 256
    img_data = np.array(img_data.resize([nw, nh]))

    # pad到 256 x 256
    if w > h:
        p = 256 - nh
        img_data = np.pad(img_data, [[p, 0], [0, 0], [0, 0]], 'reflect')
    else:
        p = 256 - nw
        img_data = np.pad(img_data, [[0, 0], [p, 0], [0, 0]], 'reflect')

    return img_data/128.0 - 1.0

def _refine2(img_path):
    img_data = Image.open(img_path)
    # resize到长边256
    w, h = img_data.size
    if w > h:
        nw, nh = 256, int(h * 256 / w)
    else:
        nw, nh = int(w * 256 / h), 256
    img_data = np.array(img_data.resize([nw, nh]))

    # pad到 256 x 256
    if w > h:
        p = (256 - nh)//2
        q = 256 - p - nh
        img_data = np.pad(img_data, [[p, q], [0, 0], [0, 0]], 'constant', constant_values=127)
    else:
        p = (256 - nw)//2
        q = 256 - p - nw
        img_data = np.pad(img_data, [[0, 0], [p, q], [0, 0]], 'constant', constant_values=127)
    
    assert(img_data.shape == (256, 256, 3))
    
    return img_data/127.0 - 1.0

# ------------verify transfer
with open(path.join(tr_dir, 'labels.json')) as f:
    tr_idx = json.load(f)

with open(path.join(v_dir, 'labels.json')) as f:
    v_idx = json.load(f)

tr_h5_file = 'F:/dataset/training.h5'
v_h5_file = 'F:/dataset/validation.h5'
test_h5_file = 'F:/dataset/test.h5'

def isValid(image_id, h5fp, y_label):
    full_lb = h5fp[image_id]['full_lb'].value
    disease_class = np.argmax(full_lb)
    if disease_class == y_label:
        return True
    else:
        return image_id

def check_training():
    test_list = []
    success_count = 0
    with h5py.File(tr_h5_file, 'r') as fp:
        for idx, item in enumerate(tr_idx):
            y_label = item['disease_class']
            img_id = item['image_id']

            try:
                tested = isValid(img_id, fp, y_label)
                success_count += 1
            except:
                test_list.append(item)

            print(f'tr verification: {idx}', end='\r')

    return success_count, test_list

def check_test(test_list):
    success_count = 0
    failure = []
    with h5py.File(test_h5_file, 'r') as fp:
        for idx, item in enumerate(test_list):
            y_label = item['disease_class']
            img_id = item['image_id']

            tested = isValid(img_id, fp, y_label)
            if tested is True:
                success_count += 1
            else:
                failure.append(item)

            print(f'test verification: {idx}', end='\r')

    return success_count, failure

def check_validation():
    success_count = 0
    failure = []
    with h5py.File(v_h5_file, 'r') as fp:
        for idx, item in enumerate(v_idx):
            y_label = item['disease_class']
            img_id = item['image_id']

            try:
                tested = isValid(img_id, fp, y_label)
                success_count += 1
            except:
                failure.append(item)

            print(f'test verification: {idx}', end='\r')

    return success_count, failure


###### bottle-neck test features
test_h5_path = 'f:/dataset/AgriculturalDisease/test.h5'

def extract_test_features(store_path=test_h5_path):
    tf.reset_default_graph()
    # gsm = tf.keras.applications.DenseNet121(
    #     include_top=False, input_shape=[256, 256, 3])
    gsm = tf.keras.applications.DenseNet201(input_shape=[256, 256, 3], include_top=False)
    files = glob('F:/dataset/AgriculturalDisease/testA/images/*')

    with h5py.File(store_path) as f:
        batch_size = 1024
        total = len(files) // 1024 + 1
        print('total:', total)
        for i in range(total):
            print(f'dealing with {i+1}', end='\r')
            img_paths = files[i*batch_size : (i+1)*batch_size]
            store_stack_features(img_paths, f, gsm)


def store_stack_features(batch_paths, h5f, gsm):
    stack_img = []
    img_indeces = []
    for p in batch_paths:
        img_id = path.basename(p)
        img_data = _refine(p)
        stack_img.append(img_data)
        img_indeces.append(img_id)
    
    stack_img = np.array(stack_img)
    features = gsm.predict(stack_img)
    
    for img_id, feat in zip(img_indeces, features):
        h5f[img_id] = feat


def flow_y_X(data_dir="trainingset", batch_size=512):
    base = path.join('F:/dataset/AgriculturalDisease', data_dir)
    with open(path.join(base, 'labels.json'), 'r') as f:
        lb_json = json.load(f)
    
    total = len(lb_json) // batch_size + 1
    print('total:', total)

    for i in range(total):
        print('dealing with', i, end='\r')
        img_batch_info = lb_json[i*batch_size : (i+1)*batch_size]

        indeces = [info['image_id'] for info in img_batch_info]
        img_paths = [path.join(base, 'images', id) for id in indeces]
        data = np.array([_refine2(p) for p in img_paths])

        labels = [info['disease_class'] for info in img_batch_info]
        labels = list(zip(labels, indeces))

        idg = ImageDataGenerator(fill_mode='constant', width_shift_range=0.15,
                                height_shift_range=0.15, rotation_range=360,
                                vertical_flip=True, horizontal_flip=True)
        flow = idg.flow(data, labels, batch_size=batch_size, shuffle=True)
        data, labels = next(idg.flow(data, labels, batch_size=batch_size))

        del idg
        img_indeces = indeces

        yield data, labels
        del data, labels
        gc.collect()


def mutate_to_h5df(data_dir="trainingset", batch_size=512, startId=0):
    store_path = path.join('F:/dataset/AgriculturalDisease', data_dir+' - augmeted.h5')
    gsm = tf.keras.applications.DenseNet121(
        include_top=False, input_shape=[256, 256, 3])
    print('gsm initialized')

    with h5py.File(store_path, 'a') as h5fp:
        flow = flow_y_X(data_dir, batch_size)
        control_id = startId
        for data, labels in flow:
            features = gsm.predict(data)
            del data
            for feat, (lb, img_id) in zip(features, labels):
                group = h5fp.create_group(str(control_id))

                group['lb'] = str(lb)
                group['img_id'] = str(img_id)
                group['feat'] = feat

                control_id += 1

            del features, labels

    return control_id