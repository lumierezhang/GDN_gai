import numpy as np
import torch
import matplotlib.pyplot as plt
import torch.nn as nn
import time
from util.time import *
from util.env import *
from torch_geometric.nn import GCNConv, GATConv, EdgeConv
import math
import torch.nn.functional as F

from models.m_lstm import LSTM_Model
from models.USAD import Usadmodel
from models.beatgan import Genertor
from .graph_layer import GraphLayer


def get_batch_edge_index(org_edge_index, batch_num, node_num):
    # org_edge_index:(2, edge_num)
    edge_index = org_edge_index.clone().detach()
    edge_num = org_edge_index.shape[1]
    batch_edge_index = edge_index.repeat(1,batch_num).contiguous()

    for i in range(batch_num):
        batch_edge_index[:, i*edge_num:(i+1)*edge_num] += i*node_num

    return batch_edge_index.long()


class OutLayer(nn.Module):
    def __init__(self, in_num, node_num, layer_num, inter_num = 512):
        super(OutLayer, self).__init__()

        # print(in_num, "outLayer_in-num") (64)
        # print(node_num, "out_node-num") (27)
        # print(layer_num, "out_layer-num") (1)
        modules = []

        for i in range(layer_num):
            # last layer, output shape:1
            if i == layer_num-1:
                modules.append(nn.Linear( in_num if layer_num == 1 else inter_num, 1))
            else:
                layer_in_num = in_num if i == 0 else inter_num
                modules.append(nn.Linear( layer_in_num, inter_num ))
                modules.append(nn.BatchNorm1d(inter_num))
                modules.append(nn.ReLU())

        self.mlp = nn.ModuleList(modules)

    def forward(self, x):
        out = x
        # print(out.shape,"输入outlayer数据") (127,27,64)

        for mod in self.mlp:
            if isinstance(mod, nn.BatchNorm1d):
                out = out.permute(0,2,1)  # 转置
                out = mod(out)
                out = out.permute(0,2,1)
            else:
                out = mod(out)
        # print(out.shape, "输出outlayer数据") (128,27,1)
        # print("2")
        return out



class GNNLayer(nn.Module):
    def __init__(self, in_channel, out_channel, inter_dim=0, heads=1, node_num=100):  # （15 64）
        super(GNNLayer, self).__init__()


        self.gnn = GraphLayer(in_channel, out_channel, inter_dim=inter_dim, heads=heads, concat=False)  #  ( in=15  out=64)将卷积算子推广到不规则域通常表示为邻域聚合或消息传递方案。

        self.bn = nn.BatchNorm1d(out_channel)  # (此时out为特征维度 归一化)
        self.relu = nn.ReLU()
        self.leaky_relu = nn.LeakyReLU()  # (Leaky ReLU是给所有负值赋予一个非零斜率)

    def forward(self, x, edge_index, embedding=None, node_num=0):
        # print(x.shape,"进入gnn层喂进来的数据x")  (3456,15)
        # print(edge_index.shape, "喂进gnn的数据edge_index") (2, 69120)
        out, (new_edge_index, att_weight) = self.gnn(x, edge_index, embedding, return_attention_weights=True)
        # print(out.shape,"处理之后的out数据") (3456,64)
        self.att_weight_1 = att_weight
        self.edge_index_1 = new_edge_index
  
        out = self.bn(out)
        # print(out.shape,"准备输出out数据") (3456,64)
        
        return self.relu(out)


class GDN(nn.Module):
    def __init__(self, edge_index_sets, node_num, dim=64, out_layer_inter_dim=256, input_dim=10, out_layer_num=1, topk=5):
        # ((2,702))
        super(GDN, self).__init__()

        self.edge_index_sets = edge_index_sets

        device = get_device()

        edge_index = edge_index_sets[0]  # (2,702)

        embed_dim = dim  # （64）
        self.embedding = nn.Embedding(node_num, embed_dim)  # 转换成对应计算机识别的数字，可以自动学习每个词向量对应的w权重
        self.bn_outlayer_in = nn.BatchNorm1d(embed_dim)  # 归一化


        edge_set_num = len(edge_index_sets)  # （1）
        self.gnn_layers = nn.ModuleList([
            GNNLayer(input_dim, dim, inter_dim=dim+embed_dim, heads=1) for i in range(edge_set_num)
        ])  # 以列表的形式来保持多个子模块


        self.node_embedding = None
        self.topk = topk
        self.learned_graph = None
        # print("2")
        self.out_layer = OutLayer(dim*edge_set_num, node_num, out_layer_num, inter_num = out_layer_inter_dim)

        self.cache_edge_index_sets = [None] * edge_set_num
        self.cache_embed_index = None

        self.dp = nn.Dropout(0.2)

        self.init_params()
        # print("***********************新加*********************************")

        # print("AE")

        # self.ll = LSTM_Model(27, 20) # (msl)
        self.ll = LSTM_Model(50, 35)  # (swat)
        # self.m = Usadmodel(w_size=405, z_size=101)  # (msl)
        # self.m = Usadmodel(w_size=15, z_size=1)
        self.m = Usadmodel(w_size = 750, z_size = 187)  # (swat)
        # self.m = Usadmodel(w_size=3810, z_size=952)  # (wadi)
        # self.beat = Genertor()
        # print("***********************新加*********************************")
    
    def init_params(self):
        nn.init.kaiming_uniform_(self.embedding.weight, a=math.sqrt(5))


    def forward(self, data, org_edge_index):

        x = data.clone().detach()  # (128,27,15)  （每次取一个batch）  (detach 网络运行保留一些参数)
        # print(x.shape,"22")
        # x = x.view(x.shape[0],15,27)  # (msl)
        x = x.view(x.shape[0],15,50)  # (swat)
        x = self.ll(x)  # （msl （128，15，27）） # 新加lstm
        # print(x.shape,"22")
        # exit(0)
        # print("***********************新加AE（全连接）*********************************")
        x = x.view(x.shape[0],-1)  # (128,750)  (15,1,3456)wwww
        # print(x.shape,"11")
        # exit(0)
        x = self.m(x)  #dddd

        # x = x.view(-1,27,15)  # (msl)
        x = x.view(-1,50,15)  # (swat)
        # x = x.view(-1,127,30)  # (wadi)
        # print("***********************新加AE（全连接）*********************************")

        edge_index_sets = self.edge_index_sets

        device = data.device

        batch_num, node_num, all_feature = x.shape  # (128,    27,    15)
        x = x.view(-1, all_feature).contiguous()  # (3456,15)
        # print(x.shape, "变形过后x数据")

        gcn_outs = []
        for i, edge_index in enumerate(edge_index_sets):
            edge_num = edge_index.shape[1]  # (702)
            cache_edge_index = self.cache_edge_index_sets[i]  # (none)

            # print(edge_num, "edge_num")
            # print(cache_edge_index, "cache_edge_index")
            if cache_edge_index is None or cache_edge_index.shape[1] != edge_num*batch_num:
                self.cache_edge_index_sets[i] = get_batch_edge_index(edge_index, batch_num, node_num).to(device)
            
            batch_edge_index = self.cache_edge_index_sets[i]
            
            all_embeddings = self.embedding(torch.arange(node_num).to(device))

            weights_arr = all_embeddings.detach().clone()  # (27,64)
            # print(all_embeddings.shape, "all_embedding")
            all_embeddings = all_embeddings.repeat(batch_num, 1)  # (3456,64)
            # print(all_embeddings.shape, "all_embedding shape")

            weights = weights_arr.view(node_num, -1)  # (27,64)
            # print(weights.shape, "weights")

            cos_ji_mat = torch.matmul(weights, weights.T)  # (27,27)  # （3.4 中公式2）
            # print(cos_ji_mat.shape,"cos_ji_mat")
            normed_mat = torch.matmul(weights.norm(dim=-1).view(-1,1), weights.norm(dim=-1).view(1,-1))  # (27,27)
            # print(normed_mat.shape,"normed_mat")
            cos_ji_mat = cos_ji_mat / normed_mat

            dim = weights.shape[-1]
            topk_num = self.topk

            topk_indices_ji = torch.topk(cos_ji_mat, topk_num, dim=-1)[1]

            self.learned_graph = topk_indices_ji

            gated_i = torch.arange(0, node_num).T.unsqueeze(1).repeat(1, topk_num).flatten().to(device).unsqueeze(0)
            gated_j = topk_indices_ji.flatten().unsqueeze(0)
            gated_edge_index = torch.cat((gated_j, gated_i), dim=0)

            batch_gated_edge_index = get_batch_edge_index(gated_edge_index, batch_num, node_num).to(device)
            # print(batch_gated_edge_index.shape, "进入gnn层之前") (2,69120)
            # print("1")

            # Graph Attention-based Forecasting
            # (1) Feature Extractor
            gcn_out = self.gnn_layers[i](x, batch_gated_edge_index, node_num=node_num*batch_num, embedding=all_embeddings)

            
            gcn_outs.append(gcn_out)

        x = torch.cat(gcn_outs, dim=1)  # (3456,64)
        # print(x.shape, "变形过后x数据")
        # exit(0)
        x = x.view(batch_num, node_num, -1)


        indexes = torch.arange(0,node_num).to(device)  # (27)
        # print(indexes.shape, "indexes shape")
        out = torch.mul(x, self.embedding(indexes))  # x= (128,27,64) out=(128,27,64)
        # print(x.shape,"x shpae")
        # print(out.shape, "first out")
        
        out = out.permute(0, 2, 1)
        out = F.relu(self.bn_outlayer_in(out))
        out = out.permute(0, 2, 1)

        out = self.dp(out)
        # Output Layer
        out = self.out_layer(out)  # (128,27,1)
        # out = torch.transpose(out,2,1)
        # print(out.shape,"out shape")
        out = out.view(-1, node_num) # (128,27)
        # print(out.shape,"处理完输出数据")

        # print("****************************************************")
        # out = out.view(out.shape[0],-1)  # (128,1,50) swat
        # out = out.view(out.shape[0],1,out.shape[1])  # (128,1,50) swat
        # print(out.shape,"处理完输出数据11")
        # out = self.beat(out)  # 新加beatgan
        # out = out.view(out.shape[0],-1)
        # print(out.shape,"处理完输出数据22")
        # exit(0)

        # print("****************************************************")


        return out



