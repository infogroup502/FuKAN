import torch
import torch.nn as nn
from tkinter import _flatten
import torch.nn.functional as F
from model.bspline import KANspline
from model.KAN import KANLinear


class fuzzy(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(fuzzy, self).__init__()
        self.KAN = KANLinear(input_size,output_size)
        self.sigma = nn.Parameter(torch.rand(input_size,hidden_size))
        self.mu = nn.Parameter(torch.rand(input_size,hidden_size))
    def forward(self, x):
        fuzzy_value = torch.exp(-((x.unsqueeze(-1)-self.mu.unsqueeze(0))**2)/(2*(self.sigma.unsqueeze(0)**2)))
        x = self.KAN(fuzzy_value.permute(0,2,1)).permute(0,2,1)
        return x,fuzzy_value


class FuKAN(nn.Module):
    def __init__(self, win_size,d_model=256, local_size=[3, 5, 7],global_size=[3,5,7], channel=55,dropout=0.05, output_attention=True,seq_len=2):
        super(FuKAN, self).__init__()
        self.output_attention = output_attention
        self.local_size = local_size
        self.channel = channel
        self.win_size = win_size
        self.hiddensize = 4
        self.global_size=global_size
        print("hidden_size",self.hiddensize)

        self.fuzzy_size_front = nn.ModuleList(
            fuzzy(localsize*seq_len, self.hiddensize, global_size[index]*seq_len) for index, localsize in enumerate(self.local_size))

        self.fuzzy_num_front = nn.ModuleList(
            fuzzy(global_size[index]*seq_len, self.hiddensize    , localsize*seq_len) for index, localsize in enumerate(self.local_size))

        self.fuzzy_size_back = nn.ModuleList(
            fuzzy(localsize*seq_len, self.hiddensize, global_size[index]*seq_len) for index, localsize in
            enumerate(self.local_size))

        self.fuzzy_num_back = nn.ModuleList(
            fuzzy(global_size[index]*seq_len, self.hiddensize, localsize*seq_len) for index, localsize in
            enumerate(self.local_size))
        self.d_model = d_model
        self.dropout = nn.Dropout(dropout)


    def forward(self, in_size,in_num,in_local_back,in_global_back):
        local_mean = []
        global_mean = []
        local_mean_fuzzy = []
        global_mean_fuzzy = []
        local_mean_back = []
        global_mean_back = []
        local_mean_fuzzy_back = []
        global_mean_fuzzy_back = []
        B, L, M, _ = in_size[0].shape

        for index, localsize in enumerate(self.local_size):
            x_local, x_global,x_local_back,x_global_back = in_size[index], in_num[index],in_local_back[index],in_global_back[index]

            x_local,x_localfuzzy = self.fuzzy_size_front[index](x_local.reshape(B*L*M,-1))
            x_global,x_globalfuzzy = self.fuzzy_num_front[index](x_global.reshape(B*L*M,-1))

            x_local_back,  x_localfuzzy_back = self.fuzzy_size_back[index](x_local_back.reshape(B * L * M, -1))
            x_global_back, x_globalfuzzy_back = self.fuzzy_num_back[index](x_global_back.reshape(B * L * M, -1))

            local_mean.append(x_local.reshape(B,L,M,-1)), global_mean.append(x_global.reshape(B,L,M,-1)),local_mean_fuzzy.append(x_localfuzzy.reshape(B,L,M,-1)),global_mean_fuzzy.append(x_globalfuzzy.reshape(B,L,M,-1))

            local_mean_back.append(x_local_back.reshape(B, L, M, -1)), global_mean_back.append(x_global_back.reshape(B, L, M, -1)), local_mean_fuzzy_back.append(x_localfuzzy_back.reshape(B, L, M, -1)), global_mean_fuzzy_back.append(x_globalfuzzy_back.reshape(B, L, M, -1))
        local_mean = list(_flatten(local_mean))  # 3
        global_mean = list(_flatten(global_mean))  # 3
        local_mean_fuzzy = list(_flatten(local_mean_fuzzy))  # 3
        global_mean_fuzzy = list(_flatten(global_mean_fuzzy))
        local_mean_back = list(_flatten(local_mean_back))  # 3
        global_mean_back = list(_flatten(global_mean_back))  # 3
        local_mean_fuzzy_back = list(_flatten(local_mean_fuzzy_back))  # 3
        global_mean_fuzzy_back = list(_flatten(global_mean_fuzzy_back))
        if self.output_attention:
            return local_mean, global_mean,local_mean_fuzzy, global_mean_fuzzy,local_mean_back, global_mean_back,local_mean_fuzzy_back, global_mean_fuzzy_back
        else:
            return None