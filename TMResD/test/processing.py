'''
# !/usr/bin/python3
# -*- coding: utf-8 -*-
@Time : 2021/12/22 18:33
@Author : Qiufen.Chen
@FileName: data_split.py
@Software: PyCharm
'''

from random import random
import torch.utils.data as Data
from torch.utils.data import DataLoader
import random
import numpy as np
import os


"""Read Data"""
class Contact():
    def save_file(self, li, save_path):
        str = '\n'
        f = open(save_path, "w")
        f.write(str.join(li))
        f.close()

    def read_csv(self):
        Id_path = "/lustre/home/qfchen/ResDistancePre/TMConD_id.txt"
        # -----------------------------------------------------------------------------------------------
        # data = pd.read_csv(Id_path, usecols=[0], encoding='utf-8', header=None, keep_default_na=False)
        # data = data.values.tolist()
        #
        # dataID = []
        # for item in data:
        #     dataID.append(item[0])
        # -----------------------------------------------------------------------------------------------
        dataID = []
        with open(Id_path, 'r') as fo:
            lines = fo.readlines()
            for line in lines:
                dataID.append(line.replace('\n', ''))

        random.seed(1023)
        random.shuffle(dataID)
        nums = len(dataID)

        # 60% Train Data
        trainID = dataID[:int(0.6 * nums)]
        self.save_file(trainID, '/lustre/home/qfchen/ResDistancePre/TMResD/dataset/trainID.txt')
        # 20%  Validation Data
        valID = dataID[int(0.6 * nums):int(0.8 * nums)]
        self.save_file(valID, '/lustre/home/qfchen/ResDistancePre/TMResD/dataset/valID.txt')
        # 20%  Test Data
        testID = dataID[int(0.8 * nums):]
        self.save_file(testID, '/lustre/home/qfchen/ResDistancePre/TMResD/dataset/testID.txt')
        print('Trian Data: %d   Validation Data: %d  Test Data: %d' % (len(trainID), len(valID), len(testID)))
        return trainID, valID, testID

    def get_feature_label(self, ID, flag):
        onehot_pair = []
        onehot = []
        hmm = []
        ccmpred = []
        profold = []

        feature = []
        label = []

        label_dir = "/lustre/home/qfchen/ResDistancePre/TMResD/new_lable/"
        for item in ID:
            label.append(label_dir + item + '.npy')

        # ----------------------------------------------------------------------
        if flag == 'onehot+ccmpred':
            onehot_dir = "/lustre/home/qfchen/ResDistancePre/TMResD/onehot/"
            ccmpred_dir = "/lustre/home/qfchen/ResDistancePre/TMResD/ccmpred/"
            for item in ID:
                onehot.append(onehot_dir + item + '.npy')
                ccmpred.append(ccmpred_dir + item + '.mat')
            feature.append(onehot)
            feature.append(ccmpred)
            return feature, label

        # ----------------------------------------------------------------------
        if flag == 'onehot+profold':
            onehot_dir = "/lustre/home/qfchen/ResDistancePre/TMResD/onehot/"
            profold_dir = "/lustre/home/qfchen/ResDistancePre/TMResD/profold/"
            for item in ID:
                onehot.append(onehot_dir + item + '.npy')
                profold.append(profold_dir + item + '.npz')
            feature.append(onehot)
            feature.append(profold)
            return feature, label

        # ----------------------------------------------------------------------
        if flag == 'onehot+ccmpred+profold':
            onehot_dir = "/lustre/home/qfchen/ResDistancePre/TMResD/onehot/"
            ccmpred_dir = "/lustre/home/qfchen/ResDistancePre/TMResD/ccmpred/"
            profold_dir = "/lustre/home/qfchen/ResDistancePre/TMResD/profold/"
            for item in ID:
                onehot.append(onehot_dir + item + '.npy')
                ccmpred.append(ccmpred_dir + item + '.mat')
                profold.append(profold_dir + item + '.npz')
            feature.append(onehot)
            feature.append(ccmpred)
            feature.append(profold)
            return feature, label

        # ----------------------------------------------------------------------
        if flag == 'hmm+ccmpred':
            hmm_dir = "/lustre/home/qfchen/ResDistancePre/TMResD/hmm/"
            ccmpred_dir = "/lustre/home/qfchen/ResDistancePre/TMResD/ccmpred/"
            for item in ID:
                hmm.append(hmm_dir + item + '.npy')
                ccmpred.append(ccmpred_dir + item + '.mat')
            feature.append(hmm)
            feature.append(ccmpred)
            return feature, label

        # ----------------------------------------------------------------------
        if flag == 'ccmpred+profold':
            profold_dir = "/lustre/home/qfchen/ResDistancePre/TMResD/profold/"
            ccmpred_dir = "/lustre/home/qfchen/ResDistancePre/TMResD/ccmpred/"
            for item in ID:
                profold.append(profold_dir + item + '.npz')
                ccmpred.append(ccmpred_dir + item + '.mat')
            feature.append(ccmpred)
            feature.append(profold)
            return feature, label

        # ----------------------------------------------------------------------
        if flag == 'hmm+profold':
            hmm_dir = "/lustre/home/qfchen/ResDistancePre/TMResD/hmm/"
            profold_dir = "/lustre/home/qfchen/ResDistancePre/TMResD/profold/"
            for item in ID:
                hmm.append(hmm_dir + item + '.npy')
                profold.append(profold_dir + item + '.npz')
            feature.append(hmm)
            feature.append(profold)
            return feature, label

        # ----------------------------------------------------------------------
        if flag == 'hmm+ccmpred+profold':
            hmm_dir = "/lustre/home/qfchen/ResDistancePre/TMResD/hmm/"
            profold_dir = "/lustre/home/qfchen/ResDistancePre/TMResD/profold/"
            ccmpred_dir = "/lustre/home/qfchen/ResDistancePre/TMResD/ccmpred/"
            for item in ID:
                hmm.append(hmm_dir + item + '.npy')
                ccmpred.append(ccmpred_dir + item + '.mat')
                profold.append(profold_dir + item + '.npz')
            feature.append(hmm)
            feature.append(ccmpred)
            feature.append(profold)
            return feature, label

        #----------------------------------------------------------------------------
        elif flag == 'onehot+hmm+ccmpred':
            onehot_pair_dir = "/lustre/home/qfchen/ResDistancePre/TMResD/onehot/"
            hmm_dir = "/lustre/home/qfchen/ResDistancePre/TMResD/hmm/"
            ccmpred_dir = "/lustre/home/qfchen/ResDistancePre/TMResD/ccmpred/"
            for item in ID:
                onehot_pair.append(onehot_pair_dir + item + '.npy')
                hmm.append(hmm_dir + item + '.npy')
                ccmpred.append(ccmpred_dir + item + '.mat')
            feature.append(onehot_pair)
            feature.append(hmm)
            feature.append(ccmpred)
            return feature, label

    def main(self, flag):
        """
        :return:Train data of path, Validation data of path,Test data of path
        """
        train, val, test = self.read_csv()
        x_train, y_train = self.get_feature_label(train, flag)
        x_val, y_val = self.get_feature_label(val, flag)
        x_test, y_test = self.get_feature_label(test, flag)
        return x_train, y_train, x_val, y_val, x_test, y_test