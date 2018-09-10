import numpy as np
import tensorflow as tf
from os import path
from keras.preprocessing.image import ImageDataGenerator

base_dir = '../../dataset/AgriculturalDisease'
dir_name_list = ['testA', 'trainingset', 'validationset']

dataset = {}
for d_name in dir_name_list:
  img_dir = path.join(base_dir, d_name, 'images')
  labels_path = path.join(base_dir, d_name, 'labels.json')


class DataFlow:

  def __init__(self, batch_size=256):
    pass

  def reset(self):
    pass
  
  def feed(self):
    pass
  
  def _refine(self, img_data):
    ''' 经过检查，所有图片均长+宽均>=512，至少有一边是大于256的，对此进行缩放并做pad，
        img_data必须是pil的Image.open返回的对象，函数最终输出numpy数组 '''

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
      img_data = np.pad(img_data, [[p, 0], [0, 0]], 'reflect')
    else:
      p = 256 - nw
      img_data = np.pad(img_data, [[0, 0], [p, 0]], 'reflect')
    
    return img_data
