import torch
from torch.utils.data import Dataset, DataLoader

import torch.nn.functional as F
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import numpy as np


class TimeDataset(Dataset):
    def __init__(self, raw_data, edge_index, mode='train', config = None):
        # （edge_index(2,702)\raw_data(28列 每列1565条数据)）
        self.raw_data = raw_data

        self.config = config
        self.edge_index = edge_index
        self.mode = mode

        x_data = raw_data[:-1]  #（27列 每列1565条数据 除去lable）
        labels = raw_data[-1]


        data = x_data  # （27列 每列1565条数据）

        # to tensor
        data = torch.tensor(data).double()  # （转换成tensor目的 既变换成27*1565）
        labels = torch.tensor(labels).double()  # （shape为1565*1）

        self.x, self.y, self.labels = self.process(data, labels)
    
    def __len__(self):
        return len(self.x)


    def process(self, data, labels):
        x_arr, y_arr = [], []
        labels_arr = []

        slide_win, slide_stride = [self.config[k] for k
            in ['slide_win', 'slide_stride']
        ]  # （取出来赋值）
        is_train = self.mode == 'train'

        node_num, total_time_len = data.shape  # （node=27 total=1565）

        rang = range(slide_win, total_time_len, slide_stride) if is_train else range(slide_win, total_time_len)
        # （range= （1565-15）/5=310次）
        for i in rang:  # （每次取15个 step=5 ）

            ft = data[:, i-slide_win:i]  # （每次27*15）
            tar = data[:, i] # （15*1）

            x_arr.append(ft)  # （第一次数组0的位置存27*15）
            y_arr.append(tar)  # （同理0存15*1）

            labels_arr.append(labels[i])
        # （最终x_arr为有310个数据每个位置存27*15 y_arr有310个数据27*1 ）

        x = torch.stack(x_arr).contiguous()   # （370，27，15） # （首先拷贝了一份张量在内存中的地址，然后将地址按照形状改变后的张量的语义进行排列）
        y = torch.stack(y_arr).contiguous()  # （310，27）

        labels = torch.Tensor(labels_arr).contiguous()  # （310*1）

        return x, y, labels  # （310，27，15） （310，27） （310，1）

    def __getitem__(self, idx):

        feature = self.x[idx].double()
        y = self.y[idx].double()

        edge_index = self.edge_index.long()

        label = self.labels[idx].double()

        return feature, y, label, edge_index





