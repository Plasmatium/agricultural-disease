import numpy as np
import tensorflow as tf
from os import path
import pandas as pd

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