# Author QFIUNE
# coding=utf-8
# @Time: 2022/2/24 21:04
# @File: feature_concat.py
# @Software: PyCharm
# @contact: 1760812842@qq.com

"""Handling features"""

import math
import os
import numpy as np
import torch

# ---------------------------------------------------------------
def get_hhm(hhm_file):
    result = []
    data = np.load(hhm_file)
    for a in data:
        for b in data:
            row = np.concatenate((a, b))
            result.append(row)
    feature = np.array(result)
    width = feature.shape[0]
    width = int(math.sqrt(width))
    depth = feature.shape[1]
    feature = feature.reshape([depth, width, width])
    feature = torch.from_numpy(feature)
    return feature

# ---------------------------------------------------------------
def get_onehot_pair(onehot_pair_file):
    data = np.load(onehot_pair_file)
    feature = np.array(data)
    width = feature.shape[0]
    height = feature.shape[1]
    depth = feature.shape[2]
    feature = feature.reshape([depth, width, height])
    feature = torch.from_numpy(feature)
    return feature

# ---------------------------------------------------------------
def get_onehot(onehot_file):
    data = np.load(onehot_file)
    result = []
    for a in data:
        for b in data:
            row = np.concatenate((a, b))
            result.append(row)
    feature = np.array(result)
    width = feature.shape[0]
    width = int(math.sqrt(width))
    depth = feature.shape[1]
    feature = feature.reshape([depth, width, width])
    feature = torch.from_numpy(feature)
    return feature

# ---------------------------------------------------------------
def get_ccmpred(ccmpred_file):
    data = np.loadtxt(ccmpred_file)
    feature = np.array(data)
    width = feature.shape[0]
    height = feature.shape[1]
    depth = 1
    feature = feature.reshape([depth, width, height])
    feature = torch.from_numpy(feature)
    return feature

# ---------------------------------------------------------------
def get_profold(profold_file):
    data = np.load(profold_file)
    feature = np.array(data['cbcb'])
    width = feature.shape[0]
    height = feature.shape[1]
    depth = feature.shape[2]
    feature = feature.reshape([depth, width, height])
    feature = torch.from_numpy(feature)
    return feature

def get_lable(lable_file):
    label_data = np.load(lable_file)
    max_value = 200
    new_label = np.zeros([label_data.shape[0], label_data.shape[1]], float)
    # Min-Max Normalization
    for i in range(label_data.shape[0]):
        for j in range(label_data.shape[1]):
            new_label[i][j] = label_data[i][j] / max_value
    width, height, depth = new_label.shape[0], new_label.shape[1], 1
    new_label = new_label.reshape([depth, width, height])
    new_label = torch.from_numpy(new_label)
    return new_label

"""Test"""
#==============================================================================
def main(flag):
    hmm_dir = "/lustre/home/qfchen/ContactMap/SurContact/hmm/"
    ccmpred_dir = "/lustre/home/qfchen/ContactMap/SurContact/ccmpred/"
    id_file = "/lustre/home/qfchen/ContactMap/surConD_id.txt"

    with open(id_file) as fo:
        lines = fo.readlines()
        for line in lines:
            pdb_id = line.replace('\n', '')
            hmm_file = os.path.join(hmm_dir, pdb_id + '.npy')
            ccmpred_file = os.path.join(ccmpred_dir, pdb_id + '.mat')

            if flag == 'hmm+ccmpred':
                hmm = get_hhm(hmm_file)
                ccmpred = get_ccmpred(ccmpred_file)
                feature = torch.cat(torch.cat([hmm, ccmpred], dim=0))
                print(feature.shape)
                return feature

# ---------------------------------------------------------------
if __name__ == "__main__":
    main('hmm+ccmpred')

