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

    def __init__(self, feature_path):
        self.images = feature_path

    def __getitem__(self, index):
        if len(self.images) == 2:
            pdb_id = self.images[0][index].split('/')[-1][:6]
            print(pdb_id)

            if 'hmm' in self.images[0][index] and 'ccmpred' in self.images[1][index]:
                hmm_path = self.images[0][index]
                ccmpred_path = self.images[1][index]

                hmm = feature_concat.get_hhm(hmm_path)
                ccmpred = feature_concat.get_ccmpred(ccmpred_path)

                feature = torch.cat([hmm, ccmpred], dim=0)
                return feature, pdb_id

    def __len__(self):
        return len(self.images[0])
