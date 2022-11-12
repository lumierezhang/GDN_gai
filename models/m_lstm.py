import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size):  # (612,1200)
        super().__init__()
        self.lstm = nn.LSTM(input_size,hidden_size, num_layers=2, dropout=0.1,batch_first=True)  # (51,26)


    def forward(self, w,batch_size):
        # h0 = torch.zeros(2,batch_size,20).to(w.device)  # (msl)(h0中第三维数据为hidden_size)
        # c0 = torch.zeros(2,batch_size,20).to(w.device)
        h0 = torch.zeros(2,batch_size,35).to(w.device)  # (swat)
        c0 = torch.zeros(2,batch_size,35).to(w.device)
        x , (hi,ce)= self.lstm(w,(h0,c0)) # x = (612,12,100) hi = (2,612,100) ce = (2,612,100)

        return (hi,ce)


class Decoder(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.lstm = nn.LSTM(input_size , hidden_size, num_layers = 2, batch_first=True, dropout=0.1)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(hidden_size, input_size)

    def forward(self, z,hidden):
        output, (hidden, cell) = self.lstm(z, hidden)
        prediction = self.fc(output)

        return prediction, (hidden, cell)


class LSTM_Model(nn.Module):
    def __init__(self,w,hidden_size):
        super().__init__()
        self.encoder = Encoder(w,hidden_size)
        self.decoder1 = Decoder(w,hidden_size)
        # self.decoder2 = Decoder(w_size,hidden_size)


    def forward(self, batch):
        batch_size, sequence_length, var_length = batch.size()  # var_length = 51 batch_size = 612 sequence = 12
        z = self.encoder(batch,batch_size)  # ((2,612,26)\(2,612,26))
        inv_idx = torch.arange(sequence_length - 1, -1, -1).long()  # (12)
        w1 = []
        # w2 = []
        temp_input = torch.zeros((batch_size, 1, var_length), dtype=torch.float).to(batch.device)  # (612,1,51)
        hidden = z
        for t in range(sequence_length):
            temp_input, hidden = self.decoder1(temp_input, hidden)
            # temp_input, hidden = self.decoder2(temp_input, hidden)
            w1.append(temp_input)
            # w2.append(temp_input)
        reconstruct_output1 = torch.cat(w1, dim=1)[:, inv_idx, :]  # (612,12,51)
        # reconstruct_output2 = torch.cat(w2, dim=1)[:, inv_idx, :]  # (612,12,51)

        # z2 = self.encoder(reconstruct_output1,batch_size)  # ((2,612,26)\(2,612,26))
        # bs, sq, vl = reconstruct_output1.size()  # (bs = 612 \sq = 12\vl = 51)
        # ii = torch.arange(sq - 1, -1, -1).long()  # (12)
        # w3 = []
        # tt = torch.zeros((bs, 1, vl), dtype=torch.float).to(batch.device)  # (612,1,51)
        # hidden = z2
        # for t in range(sq):
        #     tt, hidden = self.decoder2(tt, hidden)
        #     w3.append(temp_input)
        # reconstruct_output3 = torch.cat(w3, dim=1)[:, ii, :]  # (612,12,51)
        return reconstruct_output1

