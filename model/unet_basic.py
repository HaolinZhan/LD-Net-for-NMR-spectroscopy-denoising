import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import os

warnings.filterwarnings("ignore")


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


class Model(nn.Module):  # LD-12
    def __init__(self, n_layers=12, channels_interval=24):
        super(Model, self).__init__()

        self.n_layers = n_layers
        self.channels_interval = channels_interval
        encoder_in_channels_list = [1] + [i * self.channels_interval for i in range(1, self.n_layers)]
        encoder_out_channels_list = [i * self.channels_interval for i in range(1, self.n_layers + 1)]
        # encoder_in_channels_list = [1, 64, 128, 256, 512, 512, 512, 512, 512, 512, 512, 512]
        # encoder_out_channels_list = [64, 128, 256, 512, 512, 512, 512, 512, 512, 512, 512, 512]

        # 1    => 2    => 3    => 4    => 5    => 6   => 7   => 8   => 9  => 10 => 11 =>12
        # 16384 => 8192 => 4096 => 2048 => 1024 => 512 => 256 => 128 => 64 => 32 => 16 =>  8 => 4
        self.encoder = nn.ModuleList()
        for i in range(self.n_layers):
            self.encoder.append(
                DownSamplingLayer(
                    channel_in=encoder_in_channels_list[i],
                    channel_out=encoder_out_channels_list[i]
                )
            )

        self.middle = nn.Sequential(
            nn.Conv1d(self.n_layers * self.channels_interval, self.n_layers * self.channels_interval, 15, stride=1, padding=7),
            # nn.Conv1d(512, 512, 15, stride=1, padding=7),
            nn.BatchNorm1d(self.n_layers * self.channels_interval),
            nn.LeakyReLU(negative_slope=0.01, inplace=True)
            # nn.ReLU(inplace=True)
        )

        decoder_in_channels_list = [(2 * i + 1) * self.channels_interval for i in range(1, self.n_layers)] + [
            2 * self.n_layers * self.channels_interval]
        decoder_in_channels_list = decoder_in_channels_list[::-1]
        decoder_out_channels_list = encoder_out_channels_list[::-1]
        # decoder_in_channels_list = [1024, 1024, 1024, 1024, 1024, 1024, 1536, 1536, 1536, 1280, 384, 192]
        # decoder_out_channels_list = [512, 512, 512, 512, 512, 1024, 1024, 1024, 1024, 256, 128, 64]
        self.decoder = nn.ModuleList()
        for i in range(self.n_layers):
            self.decoder.append(
                UpSamplingLayer(
                    channel_in=decoder_in_channels_list[i],
                    channel_out=decoder_out_channels_list[i],
                )
            )

        self.out = nn.Sequential(
            nn.Conv1d(1 + self.channels_interval, 1, kernel_size=1, stride=1),
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

        o = self.middle(o)

        # Up Sampling
        for i in range(self.n_layers):
            # get_feature(o)
            # [batch_size, T * 2, channels]
            o = F.interpolate(o, scale_factor=2, mode="linear", align_corners=True)
            # Skip Connection
            # diff = torch.tensor(o.size()[2] - tmp[self.n_layers - i - 1].size()[2])
            # tmp[self.n_layers - i - 1] = F.pad(tmp[self.n_layers - i - 1], (diff//2, diff//2))
            # get_feature(tmp[self.n_layers - i - 1])
            o = torch.cat([o, tmp[self.n_layers - i - 1]], dim=1)
            o = self.decoder[i](o)

        o = torch.cat([o, input], dim=1)
        o = self.out(o)
        return o

