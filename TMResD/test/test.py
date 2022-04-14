'''
# !/usr/bin/python3
# -*- coding: utf-8 -*-
@Time : 2021/12/24 14:25
@Author : Qiufen.Chen
@FileName: ccmpred_train.py
@Software: PyCharm
'''

import os
import torch
import torch as th
from torch.utils.data import DataLoader
import processing
import feature_concat
import dataset
from torch import nn
import torch.utils.data as Data
from resnet import ResNet18
import numpy as np
from itertools import chain
import math

project_path = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
print(project_path)

save_path = project_path + '/TMResD/output/'

con = processing.Contact()
x_train, y_train, x_val, y_val, x_test, y_test = con.main('hmm+ccmpred')

test = dataset.mydataset(x_test, y_test)
test_loader = DataLoader(test, batch_size=1, shuffle=False, num_workers=1)

def main(n):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(1023)

    model_path = '/lustre/home/qfchen/ResDistancePre/TMResD/model/18_hmm_ccmpred_90'
    model = ResNet18(n)
    model.load_state_dict(torch.load(model_path, map_location='cpu'))

    model.eval()
    for step, (x, y_true, pdb_id) in enumerate(test_loader):
        x = x.to(torch.float32).to(device)
        y_pred = model(x)
        y_pred = y_pred.reshape([y_pred.shape[2], y_pred.shape[3]]).detach().numpy()
        np.savetxt(save_path + pdb_id + ".mat", y_pred*200)

# --------------------------------------------------------------------------------------
if __name__ == '__main__':
    n = 61
    main(n)