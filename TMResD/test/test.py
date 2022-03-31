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

save_path = '/lustre/home/qfchen/ResDistancePre/TMResD/output/'

# ----------------------------------------------------------------------------
def pcc(y_pred,y_true):
    """caculate pcc"""
    y_true = y_true.reshape(-1).detach().numpy()
    y_pred = y_pred.reshape(-1).detach().numpy()
    cc = np.corrcoef(y_true, y_pred)
    return cc[0][1]


def mask(y_true, y_pred):
    y_true = torch.squeeze(y_true)
    y_pred = torch.squeeze(y_pred)

    mask = y_true > 0
    y_true = torch.masked_select(y_true, mask)
    y_pred = torch.masked_select(y_pred, mask)
    return y_true, y_pred

# --------------------------------------------------------------------------------------
batch_size = 1
epochs = 100


con = processing.Contact()
# x_train, y_train, x_val, y_val, x_test, y_test = con.main('ccmpred+profold')
# x_train, y_train, x_val, y_val, x_test, y_test = con.main('hmm+ccmpred+profold')
# x_train, y_train, x_val, y_val, x_test, y_test = con.main('onehot+ccmpred')
# x_train, y_train, x_val, y_val, x_test, y_test = con.main('hmm+ccmpred')
x_train, y_train, x_val, y_val, x_test, y_test = con.main('onehot+hmm+ccmpred')
print(len(x_test[0]))
test = dataset.mydataset(x_test, y_test)
test_loader = DataLoader(test, batch_size=1, shuffle=False, num_workers=1)


def main(n):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(1023)

    w_loss, w_mae, w_mse, w_r2, w_cc = 0, 0, 0, 0, 0
    criterion = nn.SmoothL1Loss(reduction='mean')
    MAE_fn = nn.L1Loss(reduction='mean')
    MSE_fn = nn.MSELoss(reduction='mean')


    # model_path = '/lustre/home/qfchen/ResDistancePre/TMResD/model/18_ccmpred_profold_90'
    # model_path = '/lustre/home/qfchen/ResDistancePre/TMResD/model/18_hmm_ccmpred_profold_90'
    # model_path = '/lustre/home/qfchen/ResDistancePre/TMResD/model/18_onehot_ccmpred_90'
    # model_path = '/lustre/home/qfchen/ResDistancePre/TMResD/model/18_hmm_ccmpred_90'
    model_path = '/lustre/home/qfchen/ResDistancePre/TMResD/model/18_onehot_hmm_ccmpred90'

    model = ResNet18(n)
    model.load_state_dict(torch.load(model_path, map_location='cpu'))

    model.eval()
    for step, (x, y_true, pdb_id) in enumerate(test_loader):
        print(x.shape, y_true.shape)
        x = x.to(torch.float32).to(device)
        y_true = y_true.to(torch.float32).to(device)
        y_pred = model(x)
        y_1 = y_true.reshape(-1).detach().numpy()
        y_2 = y_pred.reshape(-1).detach().numpy()
        np.savez(save_path + str(pdb_id).replace("'", '').replace("(", '').replace(")", '').replace(",", ''),
                 y_true = y_1, y_pred = y_2)

        y_true, y_pred = mask(y_true, y_pred)

        train_loss = criterion(y_pred, y_true)
        train_mae = MAE_fn(y_pred, y_true)
        train_mse = MSE_fn(y_pred, y_true)
        train_cc = pcc(y_pred, y_true)


        print("trian_loss: {:.4f}, trian_mae: {:.4f}, "
              "trian_mse:{:.4f}, trian_r2: {:.4f}, trian_cc: {:.4f}"
              .format(train_loss, train_mae, train_mse, train_r2, train_cc))

        # evaluating indicator
        w_mae += train_mae.detach().item()
        w_loss += train_loss.detach().item()
        w_mse += train_mse.detach().item()
        w_cc += train_cc

    w_mae /= step + 1
    w_loss /= step + 1
    w_mse /= step + 1
    w_cc /= step + 1

    print("Total_loss: {:.4f}, Total_mae: {:.4f}, "
          "Total_mse:{:.4f}, Total_r2: {:.4f}, Total_cc: {:.4f}"
          .format(w_loss, w_mae, w_mse, w_r2, w_cc))

# --------------------------------------------------------------------------------------
if __name__ == '__main__':
    n = 101
    main(n)