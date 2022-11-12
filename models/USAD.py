import torch
import torch.nn as nn
from models.Extention_att import ExternalAttention

class Encoder(nn.Module):
    def __init__(self, in_size, latent_size):
        super().__init__()
        self.EA = ExternalAttention(d_model=750, S=8)  # (swat)
        # self.EA = ExternalAttention(d_model=3810, S=8)  # (wadi)
        # self.EA = ExternalAttention(d_model=405, S=8)  # (msl)
        self.linear1 = nn.Linear(in_size, int(in_size / 2))
        self.linear2 = nn.Linear(int(in_size / 2), int(in_size / 4))
        self.linear3 = nn.Linear(int(in_size / 4), latent_size)
        self.relu = nn.ReLU(True)

    def forward(self, w):
        # print(w.shape)
        out = self.EA(w)  # (w=(128,750)swat)
        out = self.linear1(out)
        out = self.relu(out)
        out = self.linear2(out)
        out = self.relu(out)
        out = self.linear3(out)
        z = self.relu(out)
        # print(z.shape)
        return z


class Decoder(nn.Module):
    def __init__(self, latent_size, out_size):
        super().__init__()
        self.EA = ExternalAttention(d_model=187, S=2)   # (swat)
        # self.EA = ExternalAttention(d_model=952, S=2)   # (wadi)
        # self.EA = ExternalAttention(d_model=101, S=2)  # (msl)
        self.linear1 = nn.Linear(latent_size, int(out_size / 4))
        self.linear2 = nn.Linear(int(out_size / 4), int(out_size / 2))
        self.linear3 = nn.Linear(int(out_size / 2), out_size)
        self.relu = nn.ReLU(True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, z):
        out = self.EA(z)
        out = self.linear1(out)
        out = self.relu(out)
        out = self.linear2(out)
        out = self.relu(out)
        out = self.linear3(out)
        w = self.sigmoid(out)
        # print(w.shape)
        return w

class Usadmodel(nn.Module):
    # def __init__(self,w_size=405, z_size=101):  # （msl）
    def __init__(self, w_size = 750, z_size = 187):  # （swat）
    # def __init__(self, w_size=3810, z_size=952):  # （wadi）
        super().__init__()
        self.encoder = Encoder(w_size, z_size)
        self.decoder1 = Decoder(z_size, w_size)

    def forward(self,data):

        z = self.encoder(data)
        w1 = self.decoder1(z)
        # print(w1.shape,"w")

        return w1





