'''
# !/usr/bin/python3
# -*- coding: utf-8 -*-
@Time : 2021/12/24 14:25
@Author : Qiufen.Chen
@FileName: ccmpred_test.py
@Software: PyCharm
'''

import os

# os.environ["CUDA_VISIBLE_DEVICES"] = "5"
# from sklearn.preprocessing import MinMaxScaler
import torch
import torch as th
# from torchsummary import summary
# from torchvision.transforms import transforms
from torch.utils.data import DataLoader
import processing
import feature_concat
import dataset
from torch import nn
import torch.utils.data as Data
from resnet import SENet24, CNNnet
import numpy as np
from itertools import chain
import math
# scaler = MinMaxScaler(feature_range=[0, 1])

# ----------------------------------------------------------------------------
def pcc(y_pred,y_true):
    """caculate pcc"""
    y_true = y_true.reshape(-1).detach().numpy()
    y_pred = y_pred.reshape(-1).detach().numpy()
    cc = np.corrcoef(y_true, y_pred)
    return cc[0][1]

def mask(y_true, y_pred):
    # print(y_true.shape, y_pred.shape)
    y_true = torch.squeeze(y_true)
    y_pred = torch.squeeze(y_pred)
    # print(y_true.shape, y_pred.shape)

    mask = y_true > 0
    y_true = torch.masked_select(y_true, mask)
    y_pred = torch.masked_select(y_pred, mask)
    # print(y_true.shape, y_pred.shape)
    return y_true, y_pred


# ---------------------------------------------------------------------------
save_path = './output/'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(1023)


con = processing.Contact()
# 1.If the model selected is '24_ccmpred_90', select this line of code
# x_test, y_test = con.main('ccmpred')

# 2.If the model selected is '24_hhm_ccmpred_90', select this line of code
x_test, y_test = con.main('hhm+ccmpred')

# 3.If the model selected is '24_onehot_hhm_ccmpred90', select this line of code
# x_test, y_test = con.main('onehot+hhm+ccmpred')

print(len(x_test[0]))
test = dataset.mydataset(x_test, y_test)
test_loader = DataLoader(test, batch_size=1, shuffle=False, num_workers=1)


def main(n):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(1023)  

    w_loss, w_mae, w_mse, w_cc = 0, 0, 0, 0
    criterion = nn.SmoothL1Loss(reduction='mean')
    MAE_fn = nn.L1Loss(reduction='mean')
    MSE_fn = nn.MSELoss(reduction='mean')

   
    # 1.If the model selected is '24_ccmpred_90', select this line of code
    # model_path = './model/24_ccmpred_90'

    # If the model selected is '24_hhm_ccmpred_90', select this line of code
    model_path = './model/24_hhm_ccmpred_90'

    # 3.If the model selected is '24_onehot_hhm_ccmpred90', select this line of code
    # model_path = './model/24_onehot_hhm_ccmpred90'

    model = SENet24(n)
    model.load_state_dict(torch.load(model_path, map_location='cpu'),strict=True)
    
    model.eval()
    for step, (x, y_true, pdb_id) in enumerate(test_loader):
        # print(pdb_id)
        x = x.to(torch.float32).to(device)
        y_true = y_true.to(torch.float32).to(device)
        y_pred = model(x)

        y_true, y_pred = mask(y_true, y_pred)

        test_loss = criterion(y_pred, y_true)
        test_mae = MAE_fn(y_pred, y_true)
        test_mse = MSE_fn(y_pred, y_true)
        test_cc = pcc(y_pred, y_true)

        y_true = y_true.reshape(-1).detach().numpy()
        y_pred = y_pred.reshape(-1).detach().numpy()
        
        np.savez(save_path + pdb_id[0], y_true=y_true, y_pred=y_pred)
        print("test_loss: {:.4f}, test_mae: {:.4f}, "
              "test_mse:{:.4f}, test_cc: {:.4f}"
              .format(test_loss, test_mae, test_mse, test_cc))

        w_mae += test_mae.detach().item()
        w_loss += test_loss.detach().item()
        w_mse += test_mse.detach().item()
        w_cc += test_cc

    w_mae /= step + 1
    w_loss /= step + 1
    w_mse /= step + 1
    w_cc /= step + 1

    # evaluating indicator
    print("Total_loss: {:.4f}, Total_mae: {:.4f}, "
          "Total_mse:{:.4f},  Total_cc: {:.4f}"
          .format(w_loss, w_mae, w_mse, w_cc))


if __name__ == '__main__':
    # 1.If the model selected is '24_ccmpred_90', select this line of code
    # n = 1

    # 2.If the model selected is '24_hhm_ccmpred_90', select this line of code
    n = 61  

    # 3.If the model selected is '24_onehot_hhm_ccmpred90', select this line of code
    # n = 101
    main(n)