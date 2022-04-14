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

project_path = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
print(project_path)

"""Read Data"""
class Contact():
    def save_file(self, li, save_path):
        str = '\n'
        f = open(save_path, "w")
        f.write(str.join(li))
        f.close()

    def read_txt(self):
        # test_ID
        Id_path = project_path + "/TMResD_id.txt"
        dataID = []
        with open(Id_path, 'r') as fo:
            lines = fo.readlines()
            for line in lines:
                dataID.append(line.replace('\n', ''))
        testID = dataID
        return testID

    def get_feature(self, ID, flag):
        hmm = []
        ccmpred = []
        feature = []
        if flag == 'hmm+ccmpred':
            hmm_dir = project_path + "/TMResD/example/hmm/"
            ccmpred_dir = project_path + "/TMResD/example/ccmpred/"
            for item in ID:
                hmm.append(hmm_dir + item + '.npy')
                ccmpred.append(ccmpred_dir + item + '.mat')
            feature.append(hmm)
            feature.append(ccmpred)
            return feature

    def main(self, flag):
        """
        :return:Test data of path
        """
        test = self.read_txt()
        x_test = self.get_feature(test, flag)
        return x_test
