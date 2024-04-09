import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from torch import nn
import os
# import matplotlib.pyplot as plt
from scipy.signal import find_peaks


os.environ["CUDA_VISIBLE_DEVICES"] = "3"

# ones_tensor = torch.ones([32, 1, 8192])
# ones_tensor = torch.divide(ones_tensor, 1).cuda()
# sm = torch.zeros(8192)
# b = torch.ones(3000)
# sm[2500:5500] = sm[2500:5500]+b
# sm = sm.cuda()


class NMSELoss(nn.Module):
    def __init__(self):
        super(NMSELoss, self).__init__()
        return

    def forward(self, x, label):
        loss0 = nn.MSELoss(reduction='none')
        squared_difference = loss0(x, label)
        e = torch.sqrt(torch.sum(squared_difference))
        f = torch.sqrt(torch.sum(torch.square(label)))
        nmse = e/f
        return nmse


class NMSELossweaks(nn.Module):
    def __init__(self):
        super(NMSELossweaks, self).__init__()
        return

    def forward(self, x, label, weak_peaks):
        loss0 = nn.MSELoss(reduction='none')
        squared_difference = loss0(x, label)
        squared_difference = squared_difference * weak_peaks
        e = torch.sqrt(torch.sum(squared_difference))
        f = torch.sqrt(torch.sum(torch.square(label)))
        nmse = e/f
        return nmse


class NMSELoss_weakpeaks(nn.Module):
    def __init__(self):
        super(NMSELoss_weakpeaks, self).__init__()
        return

    def forward(self, x, label, weak_peaks):
        loss1 = NMSELoss()

        strong_peak_loss = loss1(x, label)
        weak_peak_loss = loss1(x*weak_peaks, label*weak_peaks)

        return 1/20*strong_peak_loss + 1*weak_peak_loss


class NMSELoss_weakpeaksplus(nn.Module):
    def __init__(self):
        super(NMSELoss_weakpeaksplus, self).__init__()
        return

    def forward(self, x, label, weak_peaks, array_coff):
        loss1 = NMSELossweaks()

        strong_peak_loss = loss1(x, label, weak_peaks, array_coff)
        weak_peak_loss = loss1(x*weak_peaks, label*weak_peaks, weak_peaks, array_coff)

        return 1/20*strong_peak_loss + 1*weak_peak_loss


class MANELoss(nn.Module):
    def __init__(self):
        super(MANELoss, self).__init__()
        return

    def forward(self, x, label):
        y_true1 = torch.add(label, ones_tensor)
        y_pred1 = torch.add(x, ones_tensor)
        a = torch.abs(torch.subtract(y_pred1, y_true1))
        a = torch.div(a, y_true1)
        a = torch.mean(a)
        return a


class SMMANELoss(nn.Module):
    def __init__(self):
        super(SMMANELoss, self).__init__()
        return

    def forward(self, x, label):
        y_true1 = torch.add(label, ones_tensor)
        y_pred1 = torch.add(x, ones_tensor)
        a = torch.abs(torch.subtract(y_pred1, y_true1))
        a = torch.div(a, y_true1)
        a = torch.mean(a)

        x = torch.mul(x, sm)
        label = torch.mul(label, sm)
        y_true2 = torch.add(label, ones_tensor)
        y_pred2 = torch.add(x, ones_tensor)
        c = torch.abs(torch.subtract(y_pred2, y_true2))
        c = torch.div(c, y_true2)
        c = torch.mean(c)

        return a + (1/80)*c


class SMNMSELoss(nn.Module):
    def __init__(self):
        super(SMNMSELoss, self).__init__()
        return

    def forward(self, x, label):
        loss0 = nn.MSELoss(reduction='none')
        squared_difference = loss0(x, label)
        e = torch.sqrt(torch.sum(squared_difference))
        f = torch.sqrt(torch.sum(torch.square(label)))
        nmse = e / f

        loss1 = nn.MSELoss(reduction='none')
        x = torch.mul(x, sm)
        label = torch.mul(label, sm)
        squared_difference = loss1(x, label)
        e = torch.sqrt(torch.sum(squared_difference))
        f = torch.sqrt(torch.sum(torch.square(label)))
        smnmse = e / f

        return nmse + 1/80*smnmse


class NmrDatasetTxT(Dataset):
    def __init__(self, path):
        self.path = path
        with open(self.path, 'r') as f:
            data = f.readlines()
            data = data[0].split()
        f.close()
        self.data = data

    def __getitem__(self, item):
        path = self.data[item]
        path = path[1:-2]
        with open(path, 'r') as f:
            raw = f.readlines()
            c = list()
            for doc in raw:
                b = doc.split()
                for i in range(len(b)):
                    c.append(float(b[i]))
                trg_data = c[:8192]
                src_data = c[8192:]
                # trg_data = c[:1024]
                # src_data = c[1024:]
        f.close()
        src_data = np.array(src_data)
        trg_data = np.array(trg_data)
        src_data = np.expand_dims(src_data, axis=0)
        trg_data = np.expand_dims(trg_data, axis=0)
        return src_data, trg_data

    def __len__(self):

        return len(self.data)


class NmrDatasetTxTweakpeaks(Dataset):
    def __init__(self, path):
        self.path = path
        with open(self.path, 'r') as f:
            data = f.readlines()
            data = data[0].split()
        f.close()
        self.data = data

    def __getitem__(self, item):
        path = self.data[item]
        path = path[1:-2]
        with open(path, 'r') as f:
            raw = f.readlines()
            c = list()
            for doc in raw:
                b = doc.split()
                for i in range(len(b)):
                    c.append(float(b[i]))
                trg_data = c[:8192]
                src_data = c[8192:]
                # trg_data = c[:1024]
                # src_data = c[1024:]
        f.close()
        peaks1, height = find_peaks(trg_data, height=0.01)
        peaks2, _ = find_peaks(trg_data, height=0.3)
        weak_peaks = [x for x in peaks1 if x not in peaks2]
        weak_peaks = np.array(weak_peaks)
        array_zeros = np.zeros(8192)
        array_ones = np.ones(8192)
        src_data = np.array(src_data)
        trg_data = np.array(trg_data)
        if len(weak_peaks):
            for x in weak_peaks:
                array_zeros[x-20:x+20] = array_ones[x-20:x+20]
        weak_peaks = array_zeros
        src_data = np.expand_dims(src_data, axis=0)
        trg_data = np.expand_dims(trg_data, axis=0)
        weak_peaks = np.expand_dims(weak_peaks, axis=0)

        return src_data, trg_data, weak_peaks

    def __len__(self):

        return len(self.data)


class NmrDatasetTxTweaks(Dataset):
    def __init__(self, path):
        self.path = path
        with open(self.path, 'r') as f:
            data = f.readlines()
            data = data[0].split()
        f.close()
        self.data = data

    def __getitem__(self, item):
        path = self.data[item]
        path = path[1:-2]
        with open(path, 'r') as f:
            raw = f.readlines()
            c = list()
            for doc in raw:
                b = doc.split()
                for i in range(len(b)):
                    c.append(float(b[i]))
                trg_data = c[:8192]
                src_data = c[8192:]
                # trg_data = c[:1024]
                # src_data = c[1024:]
        f.close()
        peaks1, height = find_peaks(trg_data, height=0.001)
        height = height['peak_heights']
        coff = 1/height
        weak_peaks = [x for x in peaks1]
        weak_peaks = np.array(peaks1)
        array_coff = np.ones(8192)
        array_ones = np.ones(8192)
        src_data = np.array(src_data)
        trg_data = np.array(trg_data)
        if len(weak_peaks):
            i = 0
            for x in weak_peaks:
                array_coff[x-20:x+20] = coff[i]
                i = i + 1
        weak_peaks = array_coff
        src_data = np.expand_dims(src_data, axis=0)
        trg_data = np.expand_dims(trg_data, axis=0)
        weak_peaks = np.expand_dims(weak_peaks, axis=0)

        return src_data, trg_data, weak_peaks

    def __len__(self):

        return len(self.data)


class NmrDatasetTxTweakpeaksplus(Dataset):
    def __init__(self, path):
        self.path = path
        with open(self.path, 'r') as f:
            data = f.readlines()
            data = data[0].split()
        f.close()
        self.data = data

    def __getitem__(self, item):
        path = self.data[item]
        path = path[1:-2]
        with open(path, 'r') as f:
            raw = f.readlines()
            c = list()
            for doc in raw:
                b = doc.split()
                for i in range(len(b)):
                    c.append(float(b[i]))
                trg_data = c[:8192]
                src_data = c[8192:]
                # trg_data = c[:1024]
                # src_data = c[1024:]
        f.close()
        peaks1, height = find_peaks(trg_data, height=0.001)
        peaks2, _ = find_peaks(trg_data, height=0.3)
        height = height['peak_heights']
        coff = 1 / height
        weak_peaks = [x for x in peaks1 if x not in peaks2]
        weak_peaks = np.array(weak_peaks)
        array_zeros = np.zeros(8192)
        array_ones = np.ones(8192)
        array_coff = np.ones(8192)
        src_data = np.array(src_data)
        trg_data = np.array(trg_data)
        if len(weak_peaks):
            i = 0
            for x in weak_peaks:
                array_zeros[x-20:x+20] = array_ones[x-20:x+20]
                array_coff[x-20:x+20] = coff[i]
                i = i + 1
        weak_peaks = array_zeros
        src_data = np.expand_dims(src_data, axis=0)
        trg_data = np.expand_dims(trg_data, axis=0)
        weak_peaks = np.expand_dims(weak_peaks, axis=0)
        array_coff = np.expand_dims(array_coff, axis=0)

        return src_data, trg_data, weak_peaks, array_coff

    def __len__(self):

        return len(self.data)


class NmrDatasetTxTfid(Dataset):
    def __init__(self, path):
        self.path = path
        with open(self.path, 'r') as f:
            data = f.readlines()
            data = data[0].split()
        f.close()
        self.data = data

    def __getitem__(self, item):
        path = self.data[item]
        path = path[1:-2]
        with open(path, 'r') as f:
            raw = f.readlines()
            c = list()
            b = raw[0].split(',')
            for i in range(len(b)):
                c.append(float(b[i]))
            trg_data_r = c[:8192]
            trg_data_i = c[8192:16384]
            src_data_r = c[16384:24576]
            src_data_i = c[24576:]

        f.close()
        src_data_r = np.array(src_data_r)
        trg_data_r = np.array(trg_data_r)
        src_data_i = np.array(src_data_i)
        trg_data_i = np.array(trg_data_i)

        src_data = np.empty((2, 8192), dtype=float)
        trg_data = np.empty((2, 8192), dtype=float)

        src_data[0][:] = src_data_r
        src_data[1][:] = src_data_i
        trg_data[0][:] = trg_data_r
        trg_data[1][:] = trg_data_i

        return src_data, trg_data

    def __len__(self):
        return len(self.data)


class NmrDatasetTxTfid_v2(Dataset):
    def __init__(self, path):
        self.path = path
        with open(self.path, 'r') as f:
            data = f.readlines()
            data = data[0].split()
        f.close()
        self.data = data

    def __getitem__(self, item):
        path = self.data[item]
        path = path[1:-2]
        with open(path, 'r') as f:
            raw = f.readlines()
            c = list()
            b = raw[0].split(',')
            for i in range(len(b)):
                c.append(float(b[i]))
            # trg_data_r = c[:8192]
            # trg_data_i = c[8192:16384]
            # src_data_r = c[16384:24576]
            # src_data_i = c[24576:]
            trg_data = c[:16384]
            src_data = c[16384:]
        f.close()
        src_data = np.array(src_data)
        trg_data = np.array(trg_data)
        src_data = np.expand_dims(src_data, axis=0)
        trg_data = np.expand_dims(trg_data, axis=0)

        return src_data, trg_data

    def __len__(self):
        return len(self.data)


class NmrDatasetTxTifft(Dataset):
    def __init__(self, path):
        self.path = path
        with open(self.path, 'r') as f:
            data = f.readlines()
            data = data[0].split()
        f.close()
        self.data = data

    def __getitem__(self, item):
        path = self.data[item]
        path = path[1:-2]
        with open(path, 'r') as f:
            raw = f.readlines()
            c = list()
            b = raw[0].split()
            for i in range(len(b)):
                c.append(float(b[i]))
            # trg_data_r = c[:8192]
            # trg_data_i = c[8192:16384]
            # src_data_r = c[16384:24576]
            # src_data_i = c[24576:]
            trg_data = c[:16384]
            src_data = c[16384:]


            # src_data_r = src_data[:8192]
        f.close()
        src_data = np.array(src_data)
        trg_data = np.array(trg_data)
        src_data = np.expand_dims(src_data, axis=0)
        trg_data = np.expand_dims(trg_data, axis=0)

        return src_data, trg_data

    def __len__(self):
        return len(self.data)


class NmrDataset(Dataset):
    def __init__(self, data=None):
        self.data = data
        self.data_lengths = len(self.data)

    def __getitem__(self, index):
        data = self.data[index]
        src_data = data[1]
        trg_data = data[0]
        src_data = torch.tensor(src_data)
        trg_data = torch.tensor(trg_data)
        src_data = src_data.type(torch.FloatTensor)
        trg_data = trg_data.type(torch.FloatTensor)
        src_data = np.expand_dims(src_data, axis=0)
        trg_data = np.expand_dims(trg_data, axis=0)
        return src_data, trg_data

    def __len__(self):
        return self.data_lengths


class NmrDataset_diff(Dataset):
    def __init__(self, data=None):
        self.data = data
        self.data_lengths = len(self.data)

    def __getitem__(self, index):
        data = self.data[index]
        src_data = data[0]
        trg_data = data[1]
        noise_data = data[2]
        src_data = torch.tensor(src_data)
        trg_data = torch.tensor(trg_data)
        noise_data = torch.tensor(noise_data)
        src_data = src_data.type(torch.FloatTensor)
        trg_data = trg_data.type(torch.FloatTensor)
        noise_data = noise_data.type(torch.FloatTensor)
        src_data = np.expand_dims(src_data, axis=0)
        trg_data = np.expand_dims(trg_data, axis=0)
        noise_data = np.expand_dims(noise_data, axis=0)
        return src_data, trg_data, noise_data

    def __len__(self):
        return self.data_lengths


def get_nmrdata(partition, args):
    if partition == 'train':
        matdata = h5py.File(args.path_train)
    elif partition == 'valid':
        matdata = h5py.File(args.path_valid)
    elif partition == 'test':
        matdata = h5py.File(args.path_test)
    else:
        raise NotImplementedError

    fft = matdata['FFT'][:].tolist()
    fftn = matdata['FFTN'][:].tolist()
    # noise = np.array(matdata['NOISE'][:])

    return fft, fftn


def get_nmrdata_diff(partition, args):
    if partition == 'train':
        matdata = h5py.File(args.path_train)
    elif partition == 'valid':
        matdata = h5py.File(args.path_valid)
    elif partition == 'test':
        matdata = h5py.File(args.path_test)
    else:
        raise NotImplementedError

    fft = matdata['FFT'][:].tolist()
    fftn = matdata['FFTN'][:].tolist()
    noise = matdata['NOISE'][:].tolist()

    # fft = np.expand_dims(fft, axis=1)
    # fftn = np.expand_dims(fftn, axis=1)
    # noise = np.float32(np.expand_dims(noise, axis=1))

    return fft, fftn, noise


def get_nmrdataset(fft, fftn, args):
    num = len(fft)
    dataset = list()
    for i in range(num):
        dataset.append((fft[i], fftn[i]))
    return dataset


def get_nmrdataset_diff(fft, fftn, noise, args):
    num = len(fft)
    dataset = list()
    for i in range(num):
        dataset.append((fft[i], fftn[i], noise[i]))
    return dataset

