# Author QFIUNE
# coding=utf-8
# @Time: 2022/2/26 20:07
# @File: dataset.py
# @Software: PyCharm
# @contact: 1760812842@qq.com


import torch
import feature_concat
import torch.utils.data as Data



# ----------------------------------------------------------------------------
class mydataset(Data.Dataset):
    """Create Dataset"""

    def __init__(self, feature_path, label_path):
        self.images = feature_path
        self.targets = label_path


    def __getitem__(self, index):
        if len(self.images) ==1:
            pdb_id = self.images[0][index].split('/')[-1][:6]
            # print(self.images[0][index], self.images[1][index])
            if 'ccmpred' in self.images[0][index]:
                ccmpred_path = self.images[0][index]
                lable_path = self.targets[index]
                ccmpred = feature_concat.get_ccmpred(ccmpred_path)
                label = feature_concat.get_lable(lable_path)
                feature = ccmpred
                return feature, label, pdb_id 

        if len(self.images) ==2:
            pdb_id = self.images[0][index].split('/')[-1][:6]
            if 'hmm' in self.images[0][index] and 'ccmpred' in self.images[1][index]:
                hmm_path = self.images[0][index]
                ccmpred_path = self.images[1][index]
                lable_path = self.targets[index]
                hmm = feature_concat.get_hhm(hmm_path)
                ccmpred = feature_concat.get_ccmpred(ccmpred_path)
                label = feature_concat.get_lable(lable_path)
                feature = torch.cat([hmm, ccmpred], dim=0)
                return feature, label, pdb_id 


        elif len(self.images) == 3:
            pdb_id = self.images[0][index].split('/')[-1][:6]
            if 'onehot' in self.images[0][index] and 'hmm' in self.images[1][index] and 'ccmpred' in self.images[2][
                index]:
                onehot_path = self.images[0][index]
                hmm_path = self.images[1][index]
                ccmpred_path = self.images[2][index]
                lable_path = self.targets[index]

                onehot = feature_concat.get_onehot(onehot_path)
                hmm = feature_concat.get_hhm(hmm_path)
                ccmpred = feature_concat.get_ccmpred(ccmpred_path)
                label = feature_concat.get_lable(lable_path)

                feature = torch.cat([torch.cat([onehot, hmm], dim=0), ccmpred], dim=0)
                # print(feature.shape)
                return feature, label, pdb_id 


    def __len__(self):
        return len(self.images[0])
