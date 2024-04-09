from torch.utils.data import dataset
import scipy.io as scio
import numpy as np
# from train import *
import matplotlib.pyplot as plt
import matplotlib
import math
from sklearn.metrics import mean_squared_error
import h5py
import os
import torch
from nmr_dataset import *
import numpy.fft as fft
import nmrglue as ng
from model.transwave import *  # LD-3
from model.unet_basic import *  # LD-12


class MyDataset(dataset.Dataset):
    def __init__(self, data=None):
        self.data = data
        self.data_lengths = len(self.data)

    def __getitem__(self, index):
        data = self.data[index]
        src_data = data[:8192]
        trg_data = data[8192:]
        src_data = torch.tensor(src_data)
        trg_data = torch.tensor(trg_data)
        src_data = src_data.type(torch.FloatTensor)
        trg_data = trg_data.type(torch.FloatTensor)
        src_data = np.expand_dims(src_data, axis=0)
        trg_data = np.expand_dims(trg_data, axis=0)
        return src_data, trg_data

    def __len__(self):
        return self.data_lengths


class MyDatasetcpu(dataset.Dataset):
    def __init__(self, data=None):
        self.data = data
        self.data_lengths = len(self.data)

    def __getitem__(self, index):
        # index = 1
        data = self.data[index]
        src_data = data[:8192]
        trg_data = data[8192:]
        # data = self.data
        # src_data = data[1024:]
        # trg_data = data[:1024]
        src_data = np.expand_dims(src_data, axis=0)
        trg_data = np.expand_dims(trg_data, axis=0)
        return src_data, trg_data

    def __len__(self):
        return self.data_lengths


class MyDataset_ftfile(dataset.Dataset):
    def __init__(self, data=None):
        self.data = data
        self.data_lengths = len(self.data)

    def __getitem__(self, index):
        data = self.data
        src_data = data
        trg_data = data
        src_data = torch.tensor(src_data)
        trg_data = torch.tensor(trg_data)
        src_data = src_data.type(torch.FloatTensor)
        trg_data = trg_data.type(torch.FloatTensor)
        # src_data = np.expand_dims(src_data, axis=0)
        # trg_data = np.expand_dims(trg_data, axis=0)
        return src_data, trg_data

    def __len__(self):
        return self.data_lengths


def get_snr(singal, freq1, freq2):
    singal = np.real(singal)
    singal_max = np.max(singal)
    noise = np.real(singal[freq1:freq2])
    snr = singal_max/np.std(noise)
    return snr


def net_test_image_ftfile(net):
    test_data = MyDataset_ftfile(datan)
    test_loader = torch.utils.data.DataLoader(dataset=test_data,
                                              batch_size=1,
                                              shuffle=True)
    net.eval()
    for step, (src_data, trg_data) in enumerate(test_loader):
        src_data = src_data.type(torch.FloatTensor)
        output = net(src_data)
    output = output.cpu()
    output = output.detach().reshape(fn)
    output = np.array(output)
    loss = mean_squared_error(output, data)
    snr1 = get_snr(np.array(src_data.cpu().detach().reshape(fn)), 8000, 8100)
    snr2 = get_snr(output, 8000, 8100)
    snr_increase = snr2 / snr1
    print("snr_increase=%.9f" % snr_increase)
    plt.figure(1), plt.plot(src_data.cpu().detach().reshape(fn))
    plt.figure(2), plt.plot(output)
    plt.figure(3), plt.plot(data)
    plt.show()


if __name__ == "__main__":
    file1 = r"/home/fangqy/PycharmProjects/LD-Net/data/simple3_nt=4.ft1"

    model_dir = r"/home/fangqy/PycharmProjects/LD-Net/12layerLD.pth"

    fn = 8192  # aqimeisu fn=32768

    dic, data = ng.pipe.read(file1)
    max_val = np.max(data)
    data = data / max_val
    datan = np.random.normal(data, 0).reshape(1, fn)

    net = torch.load(model_dir, map_location='cpu')
    net_test_image_ftfile(net)

