import numpy as np
import tensorflow as tf
import utils
import json
from glob import glob
from PIL import Image
from os import path
import time

base_dir = path.join('../../', 'dataset/AgriculturalDisease')
dir_name_list = ['testA', 'trainingset', 'validationset']

test_img_paths = glob(path.join(base_dir, 'testA', 'images', '*'))[123:123+256]


class DataFlow:

    def __init__(self, batch_size=256):
        self.batch_size = batch_size
        self.tr_dir = path.join(base_dir, 'trainingset')
        self.v_dir = path.join(base_dir, 'validationset')
        self.t_dir = path.join(base_dir, 'testA')
        self.reset()

    def init_dataIdx(self):
        file = path.join(self.tr_dir, 'labels.json')
        with open(file) as f:
            dataIdx = json.load(f)
            self.tr_dataIdx = dataIdx[:-1024]
            self.test_dataIdx = dataIdx[-1024:]

        file = path.join(self.v_dir, 'labels.json')
        with open(file) as f:
            self.v_dataIdx = json.load(f)

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
        
        total = len(dataIdx) // self.batch_size
        print(f'this epoch has total {total} batches')
        for i in range(total+1):
            meta = dataIdx[i*self.batch_size : (i+1)*self.batch_size]
            if meta == []:
                return

            img_files = [m['image_id'] for m in meta]
            raw_labels = np.array([m['disease_class'] for m in meta])

            img_paths = [path.join(img_dir, fn) for fn in img_files]
            img_data = _extract_imgs(img_paths)
            labels = utils.transfer_labels(raw_labels)
            yield img_data, labels


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
