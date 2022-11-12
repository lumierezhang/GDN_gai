import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 32, kernel_size=(4,), stride=(2,), padding=(1,), bias=False)
        self.leak = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        # self.conv2 = nn.Conv1d(32, 64, kernel_size=(4,), stride=(2,), padding=(1,), bias=False)  # (swat)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=(4,), stride=(1,), padding=(1,), bias=False)  # (wadi)(4,2)
        # self.batchn1 = nn.BatchNorm1d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        # self.conv3 = nn.Conv1d(64, 128, kernel_size=(4,), stride=(2,), padding=(1,), bias=False)
        # self.batchn2 = nn.BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        # self.conv4 = nn.Conv1d(128, 256, kernel_size=(4,), stride=(2,), padding=(1,), bias=False)
        # self.batchn3 = nn.BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        # self.conv4 = nn.Conv1d(256, 512, kernel_size=(4,), stride=(2,), padding=(1,), bias=False)
        # self.batchn4 = nn.BatchNorm1d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        # self.conv5 = nn.Conv1d(512, 50, kernel_size=(10,), stride=(1,), bias=False)
    def forward(self,z):
        out = self.conv1(z)
        out = self.leak(out)
        w = self.conv2(out)
        # out = self.batchn1(out)
        # out = self.conv3(out)
        return w


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        # self.conv1 = nn.ConvTranspose1d(50, 512, kernel_size=(10,), stride=(1,), bias=False)
        # self.batvhn1 = nn.BatchNorm1d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        # self.rr = nn.ReLU(inplace=True)
        # self.conv2 = nn.ConvTranspose1d(512, 256, kernel_size=(4,), stride=(2,), padding=(1,), bias=False)
        # self.batvhn2 = nn.BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        # self.conv3 = nn.ConvTranspose1d(256, 128, kernel_size=(4,), stride=(2,), padding=(1,), bias=False)
        # self.batvhn3 = nn.BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        # self.conv4 = nn.ConvTranspose1d(128, 64, kernel_size=(4,), stride=(2,), padding=(1,), bias=False)
        # self.batvhn4 = nn.BatchNorm1d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv5 = nn.ConvTranspose1d(64, 32, kernel_size=(4,), stride=(2,), padding=(1,), bias=False)
        self.batvhn4 = nn.BatchNorm1d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        # self.conv6 = nn.ConvTranspose1d(32, 1, kernel_size=(6,), stride=(2,), padding=(1,), bias=False)  # (swat)
        self.conv6 = nn.ConvTranspose1d(32, 1, kernel_size=(7,), stride=(2,), padding=(1,), bias=False)  # (wadi)(7,2)
        self.T = nn.Tanh()

    def forward(self,w):
        out = self.conv5(w)
        out = self.batvhn4(out)
        out = self.conv6(out)
        m = self.T(out)
        return m

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 32, kernel_size=(2,), stride=(1,), padding=(2,), bias=False)
        self.re = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=(4,), stride=(2,), padding=(1,), bias=False)
        self.bat2 = nn. BatchNorm1d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        # self.conv6 = nn.Conv1d(64,1, kernel_size=(2,), stride=(1,), bias=False)  # (swat)
        self.conv6 = nn.Conv1d(64,1, kernel_size=(1,), stride=(1,), bias=False)  # （wadi）

        # self.bat1 = nn.BatchNorm1d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        # self.conv3 = nn.Conv1d(64, 128, kernel_size=(4,), stride=(2,), padding=(1,), bias=False)
        # self.bat2 = nn. BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        # self.conv4 = nn.Conv1d(128, 256, kernel_size=(4,), stride=(2,), padding=(1,), bias=False)
        # self.bat3 = nn. BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        # self.conv5 = nn.Conv1d(256, 512, kernel_size=(4,), stride=(2,), padding=(1,), bias=False)
        # self.bat4 = nn. BatchNorm1d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        # self.conv6 = nn.Conv1d(512, 1, kernel_size=(10,), stride=(1,), bias=False)
        self.si = nn.Sigmoid()
    def forward(self,m):
        out = self.conv1(m)
        out = self.re(out)
        out = self.conv2(out)
        out = self.bat2(out)
        out = self.re(out)
        out = self.conv6(out)
        q = self.si(out)
        return  q

class Genertor(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = Encoder()
            self.decoder = Decoder()
            self.discriminator = Discriminator()



        def forward(self,data):
            # print(data.shape,"data shape")  # (128,1,50)
            out = self.encoder(data)  # (128,64,12)(swat)  (128,64,31)(wadi)
            # print(out.shape,"out11")
            out = self.decoder(out)  # (128,1,48)(swat)  (128,1,126)(wadi)
            # print(out.shape,"out22")
            out = self.discriminator(out)  # (128,1,3)(swat)  (128,2,63)(wadi)
            # print(out.shape,"out33")
            # x = self.discriminator(data)  # (128,1,3)(swat)  （wadi无）
            # print(x.shape,"x44")
            out = out.view(out.shape[0], -1)
            # x = x.view(x.shape[0], -1)  # (128,50 )(swat)
            data = data.view(data.shape[0],-1)  # (wadi/msl  中无x 直接用data替换)
            # print(data.shape,"x55")
            out = torch.add(data ,out) / 2 # (128,50)(swat)  (wadi/msl 用data代替)
            # print(out.shape,"out 77")
            # out = torch.mean(out,dim=1,keepdim=True)
            # print(out.shape,"out55")

            return out


# x = torch.ones((128,1,50))
# y = Genertor().forward(x)
# print(y.shape)