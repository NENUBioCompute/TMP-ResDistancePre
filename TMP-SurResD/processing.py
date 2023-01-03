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
    # def __init__(self):
    #     self.parent_path = os.path.abspath('..')  # get the current working parent directory
    #     print(self.parent_path)

    def save_file(self, li, save_path):
        str = '\n'
        f = open(save_path, "w")
        f.write(str.join(li))
        f.close()

    def read_csv(self):
        Id_path = "./testID.txt"
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
        testID = dataID
    
        print('Test Data: %d' % (len(testID)))

        return testID

    def get_feature_label(self, ID, flag):
        onehot = []
        hhm = []
        ccmpred = []
        feature = []
        label = []
        label_dir = "./dataset/test/lable/"
        # get lable path
        for item in ID:
            label.append(label_dir + item + '.npy')

        if flag == 'ccmpred':
            ccmpred_dir = "./dataset/test/ccmpred/"
            for item in ID:
                ccmpred.append(ccmpred_dir + item + '.mat')

            feature.append(ccmpred)
            return feature, label
 

        if flag == 'hhm+ccmpred':
            hhm_dir = "./dataset/test/hhm/"
            ccmpred_dir = "./dataset/test/ccmpred/"
            for item in ID:
                hhm.append(hhm_dir + item + '.npy')
                ccmpred.append(ccmpred_dir + item + '.mat')

            feature.append(hhm)
            feature.append(ccmpred)

            return feature, label

 
        elif flag == 'onehot+hhm+ccmpred':
            onehot_dir = "./dataset/test/onehot/"
            hhm_dir = "./dataset/test/hhm/"
            ccmpred_dir = "./dataset/test/ccmpred/"
            for item in ID:
                onehot.append(onehot_dir + item + '.npy')
                hhm.append(hhm_dir + item + '.npy')
                ccmpred.append(ccmpred_dir + item + '.mat')
                
            feature.append(onehot)
            feature.append(hhm)
            feature.append(ccmpred)

            return feature, label


    def main(self, flag):
        """
        :return:Train data of path, Validation data of path,Test data of path
        """
        test = self.read_csv()
        x_test, y_test = self.get_feature_label(test, flag)

        return x_test, y_test



class mydataset(Data.Dataset):
    def __init__(self, feature_path, label_path):
        # The definition of image's path
        self.images = feature_path
        self.targets = label_path

    def __getitem__(self, index):
        img_path, lab_path = self.images[index], self.targets[index]
        # the usage of __getitem__
        img_data = np.load(img_path)
        img_data = img_data.reshape([])
        lab_data = np.load(lab_path)
        lab_data = lab_data.reshape([])

        return img_data, lab_data

    def __len__(self):
        return len(self.images)

if __name__ == '__main__':
    con = Contact()
    x_test, y_test = con.main()
    test = mydataset(x_test, y_test)
    test_loader = DataLoader(test, batch_size=2, shuffle=False, num_workers=2)

