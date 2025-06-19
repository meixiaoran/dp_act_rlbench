import torch
import torch.nn as nn
import torch.nn.functional as F

class NLinear(nn.Module):
    """
    Normalization-Linear
    """

    def __init__(self, seq_len, pred_len, enc_in, individual=True):
        """
        初始化模型
        :param seq_len: 输入序列长度
        :param pred_len: 预测序列长度
        :param enc_in: 输入通道数
        :param individual: 是否为每个通道单独建模
        """
        super(NLinear, self).__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.dropout = nn.Dropout(p=0.1)
        self.channels = enc_in
        self.individual = individual
        # self.tcn = TemporalConv1d(in_channels=self.channels, out_channels=32, kernel_size=2, stride=1, padding=0)
        # self.deconv = nn.ConvTranspose1d(in_channels=32, out_channels=16, kernel_size=2, stride=1, padding=0)
        # self.bilstm = nn.LSTM(input_size=16, hidden_size=8, num_layers=1,
        #                                        batch_first=True, bidirectional=True)
        if self.individual:
            # 为每个通道创建独立的线性层
            self.Linear = nn.ModuleList()
            for i in range(self.channels):
                self.Linear.append(nn.Linear(self.seq_len  , self.pred_len))
        else:
            # 共享线性层
            self.Linear = nn.Linear(self.seq_len, self.pred_len)
    
        self.flatten = nn.Flatten()

        self.fc = nn.Linear(64, self.pred_len)
    def forward(self, x):
        """
        前向传播
        :param x: 输入数据，形状为 [Batch, Input length, Channel]
        :return: 输出数据，形状为 [Batch, Output length, Channel]
        """

        # 提取最后一个时间步
        # x = x.permute(0, 2, 1)
        # x = self.tcn(x)
        # x = self.deconv(x)
        # x = x.permute(0, 2, 1)

        seq_last = x[:, -1:, :].detach()
        # 去中心化
        x = x - seq_last
        # print("111", x.shape)
        if self.individual:
            # 对每个通道分别应用对应的线性层
            output = torch.zeros([x.size(0), self.pred_len,  x.size(2)], dtype=x.dtype).to(x.device)
            # print("222", output.shape)
            for i in range(self.channels):
                output[:, :, i] = self.Linear[i](x[:, :, i])
                # print("333", output.shape)
            x = output
            # print("444", output.shape)
        else:
            # 对所有通道共享一个线性层
            x = self.Linear(x.permute(0, 2, 1)).permute(0, 2, 1)

        # 加回中心
        x = x + seq_last
        # print("555", x.shape)

        # x = self.dropout(x)

        # x, y = self.attention(x, x, x)

        # attention_flatten = self.flatten(x)

        # outputs = self.fc(attention_flatten)


        return x

# def test():
#     seq_len = 2
#     pred_len = 8
#     enc_in = 8
#     individual = True

#     model = NLinear(seq_len, pred_len, enc_in, individual)
#     x = torch.randn(64, seq_len, enc_in)  # Batch size of 32
#     output = model(x)
#     print("Output shape:", output.shape)

# if __name__ == "__main__":
#     test()