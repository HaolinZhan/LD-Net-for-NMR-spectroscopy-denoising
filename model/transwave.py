import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings
import copy
from torch.nn import CrossEntropyLoss, Dropout, Softmax, Linear, Conv2d, LayerNorm
import math
from os.path import join as pjoin
import os
import matplotlib.pyplot as plt


class DownSamplingLayer(nn.Module):   #  original code
    def __init__(self, channel_in, channel_out, dilation=1, kernel_size=15, stride=1, padding=7):
        super(DownSamplingLayer, self).__init__()
        self.main = nn.Sequential(
            nn.Conv1d(channel_in, channel_out, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation),
            nn.BatchNorm1d(channel_out),
            nn.LeakyReLU(negative_slope=0.01)  # leakyRelu
            # nn.ReLU(inplace=True)
        )

    def forward(self, ipt):
        return self.main(ipt)


class UpSamplingLayer(nn.Module):
    def __init__(self, channel_in, channel_out, kernel_size=5, stride=1, padding=2):
        super(UpSamplingLayer, self).__init__()
        self.main = nn.Sequential(
            nn.Conv1d(channel_in, channel_out, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm1d(channel_out),
            nn.LeakyReLU(negative_slope=0.01, inplace=True)
            # nn.ReLU(inplace=True)
        )

    def forward(self, ipt):
        return self.main(ipt)


class ModeltransCompare(nn.Module):  # LD-3
    def __init__(self, n_layers=3, channels_interval=24):
        super(ModeltransCompare, self).__init__()

        self.n_layers = n_layers
        self.channels_interval = channels_interval

        encoder_in_channels_list = [1, 24, 48]
        encoder_out_channels_list = [24, 48, 64]

        self.encoder = nn.ModuleList()
        for i in range(self.n_layers):
            self.encoder.append(
                DownSamplingLayer(
                    channel_in=encoder_in_channels_list[i],
                    channel_out=encoder_out_channels_list[i]
                )
            )

        # self.trans = TransEncoder(config)
        self.middle = nn.Sequential(
            nn.Conv1d(64, 64, 15, stride=1, padding=7),
            # nn.Conv1d(512, 512, 15, stride=1, padding=7),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
            # nn.ReLU(inplace=True)
        )

        decoder_in_channels_list = [128, 112, 72]
        decoder_out_channels_list = [64, 48, 24]
        self.decoder = nn.ModuleList()
        for i in range(self.n_layers):
            self.decoder.append(
                UpSamplingLayer(
                    channel_in=decoder_in_channels_list[i],
                    channel_out=decoder_out_channels_list[i],
                )
            )

        self.out = nn.Sequential(
            nn.Conv1d(25, 1, kernel_size=1, stride=1),
            nn.Tanh()
        )

    def forward(self, input):
        tmp = []
        o = input
        # Down Sampling      # original code
        for i in range(self.n_layers):
            o = self.encoder[i](o)
            tmp.append(o)
            # [batch_size, T // 2, channels]
            o = o[:, :, ::2]
            # get_feature(o)

        # # Down Sampling
        # for i in range(self.n_layers):
        #     o = self.encoder[i](o)
        #     tmp.append(o)

        # o, weight = self.trans(o)

        o = self.middle(o)

        # Up Sampling
        for i in range(self.n_layers):
            # [batch_size, T * 2, channels]
            o = F.interpolate(o, scale_factor=2, mode="linear", align_corners=True)
            # Skip Connection
            # diff = torch.tensor(o.size()[2] - tmp[self.n_layers - i - 1].size()[2])
            # tmp[self.n_layers - i - 1] = F.pad(tmp[self.n_layers - i - 1], (diff//2, diff//2))

            o = torch.cat([o, tmp[self.n_layers - i - 1]], dim=1)
            o = self.decoder[i](o)

        o = torch.cat([o, input], dim=1)
        o = self.out(o)
        # print("bingo")
        return o


class Modeltrans12(nn.Module):
    def __init__(self, n_layers=11, channels_interval=24):
        super(Modeltrans12, self).__init__()

        self.n_layers = n_layers
        self.channels_interval = channels_interval

        encoder_in_channels_list = [1, 64, 256, 512, 512, 512, 512, 512, 512, 512, 512]
        encoder_out_channels_list = [64, 256, 512, 512, 512, 512, 512, 512, 512, 512, 512]

        self.encoder = nn.ModuleList()
        for i in range(self.n_layers):
            self.encoder.append(
                DownSamplingLayer(
                    channel_in=encoder_in_channels_list[i],
                    channel_out=encoder_out_channels_list[i]
                )
            )

        self.trans = TransEncoder(config)

        decoder_in_channels_list = [1280, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 512, 192]
        decoder_out_channels_list = [512, 512, 512, 512, 512, 512, 512, 512, 256, 128, 64]
        self.decoder = nn.ModuleList()
        for i in range(self.n_layers):
            self.decoder.append(
                UpSamplingLayer(
                    channel_in=decoder_in_channels_list[i],
                    channel_out=decoder_out_channels_list[i],
                )
            )

        self.out = nn.Sequential(
            nn.Conv1d(65, 1, kernel_size=1, stride=1),
            nn.Tanh()
        )

    def forward(self, input):
        tmp = []
        o = input
        # Down Sampling      # original code
        for i in range(self.n_layers):
            o = self.encoder[i](o)
            tmp.append(o)
            # [batch_size, T // 2, channels]
            o = o[:, :, ::2]

        # # Down Sampling
        # for i in range(self.n_layers):
        #     o = self.encoder[i](o)
        #     tmp.append(o)

        o, weight = self.trans(o)

        # Up Sampling
        for i in range(self.n_layers):
            # [batch_size, T * 2, channels]
            o = F.interpolate(o, scale_factor=2, mode="linear", align_corners=True)
            # Skip Connection
            # diff = torch.tensor(o.size()[2] - tmp[self.n_layers - i - 1].size()[2])
            # tmp[self.n_layers - i - 1] = F.pad(tmp[self.n_layers - i - 1], (diff//2, diff//2))

            o = torch.cat([o, tmp[self.n_layers - i - 1]], dim=1)
            o = self.decoder[i](o)

        o = torch.cat([o, input], dim=1)
        o = self.out(o)
        return o


class ModeltransNoembedding(nn.Module):
    def __init__(self, n_layers=3, channels_interval=24):
        super(ModeltransNoembedding, self).__init__()

        self.n_layers = n_layers
        self.channels_interval = channels_interval

        encoder_in_channels_list = [1, 64, 256]
        encoder_out_channels_list = [64, 256, 512]

        self.encoder = nn.ModuleList()
        for i in range(self.n_layers):
            self.encoder.append(
                DownSamplingLayer(
                    channel_in=encoder_in_channels_list[i],
                    channel_out=encoder_out_channels_list[i]
                )
            )

        self.trans = TransEncoderNoembedding(config)

        decoder_in_channels_list = [1280, 512, 128]
        decoder_out_channels_list = [256, 64, 64]
        self.decoder = nn.ModuleList()
        for i in range(self.n_layers):
            self.decoder.append(
                UpSamplingLayer(
                    channel_in=decoder_in_channels_list[i],
                    channel_out=decoder_out_channels_list[i],
                )
            )

        self.out = nn.Sequential(
            nn.Conv1d(65, 1, kernel_size=1, stride=1),
            nn.Tanh()
        )

    def forward(self, input):
        tmp = []
        o = input
        # Down Sampling      # original code
        for i in range(self.n_layers):
            o = self.encoder[i](o)
            tmp.append(o)
            # [batch_size, T // 2, channels]
            o = o[:, :, ::2]

        # # Down Sampling
        # for i in range(self.n_layers):
        #     o = self.encoder[i](o)
        #     tmp.append(o)

        o, weight = self.trans(o)

        # Up Sampling
        for i in range(self.n_layers):
            # [batch_size, T * 2, channels]
            o = F.interpolate(o, scale_factor=2, mode="linear", align_corners=True)
            # Skip Connection
            # diff = torch.tensor(o.size()[2] - tmp[self.n_layers - i - 1].size()[2])
            # tmp[self.n_layers - i - 1] = F.pad(tmp[self.n_layers - i - 1], (diff//2, diff//2))

            o = torch.cat([o, tmp[self.n_layers - i - 1]], dim=1)
            o = self.decoder[i](o)

        o = torch.cat([o, input], dim=1)
        o = self.out(o)
        return o

