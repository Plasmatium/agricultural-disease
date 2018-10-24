import numpy as np
import tensorflow as tf
from os import path
import pandas as pd
from PIL import Image

df_labels_map = pd.read_csv('labels_map.csv', index_col='Label id')
sp_classes = df_labels_map.iloc[:,1].max()+1 # 物种类别数
ds_classes = df_labels_map.iloc[:,2].max()+1 # 疾病类数
sr_classes = df_labels_map.iloc[:,3].max()+1 # 严重程度级别数
full_classes = 61
print(f'sc cls: {sp_classes}, ds cls: {ds_classes}, sr cls: {sr_classes}')

labels_map_dict = df_labels_map.to_dict()

def convert_to_one_hot(y, C):
    return np.eye(C, dtype=np.uint8)[y.reshape(-1)]


def transfer_labels(labels):
    '''将单一label转换成（物种，疾病，严重程度）三类的one-hot标签'''
    sp, ds, sr = [], [], []
    labels = labels.astype('int')
    for l in labels.reshape(-1):
        sp.append(labels_map_dict['物种'][l])
        ds.append(labels_map_dict['疾病'][l])
        sr.append(labels_map_dict['严重程度'][l])

    sp = convert_to_one_hot(np.array(sp), sp_classes)
    ds = convert_to_one_hot(np.array(ds), ds_classes)
    sr = convert_to_one_hot(np.array(sr), sr_classes)
    full_lb = convert_to_one_hot(labels, full_classes)

    return sp, ds, sr, full_lb

def show_params_scale():
    trv = np.sum([np.prod(v.shape.as_list()) for v in tf.trainable_variables()])
    allv = np.sum([np.prod(v.shape.as_list()) for v in tf.global_variables()])
    print('trainable:', trv)
    print('global:', allv)

def show_img(key, dir_='testA'):
    file = path.join('../../dataset/AgriculturalDisease/', dir_, 'images', key)
    return Image.open(file)

# 疾病标签转换，有些农作物，具有相同的病虫害，需要归到同一个病虫害标签
'''
5 -> 2 
6 -> 3
18 -> 4
27 -> 8

14 -> 13
19 -> 13

20 -> 15
21 -> 16
'''
def convert_ds_label(ds):
    pass

def bad_label(sr, sp, ds, full):
    df = df_labels_map
    cond1 = df.iloc[:,1] == sp
    cond2 = df.iloc[:,2] == ds
    cond3 = df.iloc[:,3] == sr

    rslt = df[cond1&cond2&cond3]
    # 全对
    if len(rslt) == 1 and rslt.index.values[0] == full:
        return 0
    # 对重要的两个
    if len(df[cond1&cond2]) != 0:
        return 1
    return 2


def rectify_bad_label(sr, sp, ds, full):
    df = df_labels_map
    cond1 = df.iloc[:, 1] == sp
    cond2 = df.iloc[:, 2] == ds
    cond3 = df.iloc[:, 3] == sr

    if ds == 0 and sr != 0:
        sr = 0
        cond3 = df.iloc[:, 3] = sr
        rslt = df[cond1 & cond2 & cond3].index
        if len(rslt) != 0:
            return rslt.values[0]
        else:
            return full

    if ds in [3, 16, 60] and sr != 3:
        sr = 3
        cond3 = df.iloc[:, 3] = sr
        rslt = df[cond1 & cond2 & cond3].index
        if len(rslt) != 0:
            return rslt.values[0]
        else:
            return full

    return full
