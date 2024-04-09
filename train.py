import argparse
import os
from torch import optim
# import json5
import numpy as np
import torch
from torch.utils.data import DataLoader
from util.utils import initialize_config
from run import *
from dataset.nmr_dataset import *
from model.unet_basic import *
# from model.waveunet_8 import *
# from model.unet_basic_att import *
from model.unet_basic_dn import *
# from sklearn.metrics import mean_squared_error
from torch.utils.tensorboard import SummaryWriter
from model.transwave import *

os.environ["CUDA_VISIBLE_DEVICES"] = "3"

epoch = 0


def net_train(net, criterion, optimizer, load_model=None):
    global epoch
    total_loss = 0.0
    if load_model != None:
        net = torch.load(load_model)
        criterion = NMSELoss()
        net = net.cuda()
        criterion = criterion.cuda()
        optimizer = optim.Adam(net.parameters(), lr=args.lr, weight_decay=1e-8, betas=(0.9, 0.999))
        # optimizer = optim.RMSprop(net.parameters(), lr=args.lr, weight_decay=1e-8, momentum=0.9)

    net.train()
    for step, (src_data, trg_data) in enumerate(train_loader):
        src_data = src_data.type(torch.FloatTensor)
        trg_data = trg_data.type(torch.FloatTensor)
        src_data = src_data.cuda()
        trg_data = trg_data.cuda()
        output = net(src_data)
        loss = criterion(output, trg_data)
        total_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    lr_label = 'lr_' + str(args.lr)
    bs_label = '_bs_' + str(args.bs) + '_'
    save_path = args.model_dir + lr_label + bs_label + "epoch_%d.pth" % epoch
    train_loss = total_loss / len(train_loader)
    torch.save(net, save_path)
    return save_path, train_loss


def net_val(criterion, load_model=None):
    total_loss = 0.0
    net = torch.load(load_model)
    net.eval()
    for step, (src_data, trg_data) in enumerate(valid_loader):
        src_data = src_data.type(torch.FloatTensor)
        trg_data = trg_data.type(torch.FloatTensor)
        src_data = src_data.cuda()
        trg_data = trg_data.cuda()
        output = net(src_data)
        loss = criterion(output, trg_data)
        total_loss += loss.item()
        # total_num = total_num + 1
    val_loss = total_loss / len(valid_loader)
    return val_loss


def net_optimise():
    global epoch
    worse_epochs = 0
    best_loss = 10000
    model_path = None
    best_model_path = None
    writer = SummaryWriter(args.log_dir)
    model = ModeltransCompare().cuda()
    # criterion = NMSELoss().cuda()
    # criterion = nn.L1Loss().cuda()
    criterion = NMSELoss().cuda()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-8, betas=(0.9, 0.999))
    # optimizer = optim.RMSprop(net.parameters(), lr=args.lr, weight_decay=1e-8, momentum=0.9)
    for i in range(2):
        worse_epochs = 0
        if i == 1:
            print("Finished first round of training, now entering fine-tuning stage")
            args.bs = 32
            args.lr = 3e-4
        while worse_epochs < 20:  # Early stopping on validation set after a few epochs
            # np.random.seed(66)
            print("EPOCH: " + str(epoch))
            model_path, train_loss = net_train(model, criterion, optimizer, load_model=model_path)
            curr_loss = net_val(criterion, load_model=model_path)
            writer.add_scalars('loss', {'train_loss': train_loss, 'val_loss': curr_loss}, epoch)
            epoch += 1
            if curr_loss < best_loss:
                worse_epochs = 0
                print("Performance on validation set improved from " + str(best_loss) + " to " + str(curr_loss))
                best_model_path = model_path
                best_loss = curr_loss
            else:
                worse_epochs += 1
                print("Performance on validation set worsened to " + str(curr_loss))
    print("TRAINING FINISHED - TESTING NOW AVAILABLE WITH BEST MODEL " + best_model_path)
    test_loss = net_test(best_model_path)
    writer.close()
    return best_model_path, test_loss


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default="WaveUnet", help="task name")
    # parser.add_argument('--path_train', type=str, default="/home/fangqy/simulation_data_8192_20_0-1.mat", help="train dataset path")
    # parser.add_argument('--path_valid', type=str, default="/home/fangqy/simulation_data_8192_20_0-1.mat", help="valid dataset path")
    # parser.add_argument('--path_train', type=str, default="/home/fangqy/matlab/train_data/60000_MS_snr8e52e4_sw8000_r0.3/validlist.txt", help="train dataset path")
    # parser.add_argument('--path_valid', type=str, default="/home/fangqy/matlab/train_data/60000_MS_snr8e52e4_sw8000_r0.3/testlist.txt", help="valid dataset path")
    parser.add_argument('--path_train', type=str, default="/home/fangqy/simulation_data_SS_8192_20000_snr0.10.5.mat", help="train dataset path")
    parser.add_argument('--path_valid', type=str, default="/home/fangqy/simulation_data_SS_8192_4000_snr0.10.5.mat", help="valid dataset path")
    parser.add_argument('--path_test', type=str, default="/home/fangqy/simulation_data_8192_test_0-1_1.mat", help="test dataset path")
    parser.add_argument('--fn', type=int, default=8192, help="frequency of FFT")
    parser.add_argument('--bs', type=int, default=32, help="batch size")
    parser.add_argument('--lr', type=float, default=0.001, help="learning rate")
    parser.add_argument('--model_dir', type=str, default="/home/fangqy/PycharmProjects/waveunet/save_model_dataset/experiment_10000_MS_snr8e52e4_sw8000_r0.3/", help="model save path")
    parser.add_argument('--log_dir', type=str, default="/home/fangqy/PycharmProjects/waveunet/logs_dataset/experiment_10000_MS_snr8e52e4_sw8000_r0.3/", help="log save path")
    # parser.add_argument('--model_dir', type=str, default="/home/fangqy/PycharmProjects/waveunet/save_model/experiment_test", help="model save path")
    # parser.add_argument('--log_dir', type=str, default="/home/fangqy/PycharmProjects/waveunet/save_model/experiment_test", help="log save path")
    parser.add_argument('--partition', type=str, default=['train', 'valid', 'test'], help="dataset partition")
    args = parser.parse_args()

    train_data = NmrDatasetTxT(args.path_train)
    val_data = NmrDatasetTxT(args.path_valid)

    train_loader = DataLoader(dataset=train_data, batch_size=args.bs, num_workers=4, shuffle=True)
    valid_loader = DataLoader(dataset=val_data, batch_size=args.bs, num_workers=4, shuffle=True)

    # for key in args.partition:
    #     fft, fftn = get_nmrdata(key, args)
    #     if key == 'train':
    #         train_data = NmrDataset(get_nmrdataset(fft, fftn, args))
    #     elif key == 'valid':
    #         val_data = NmrDataset(get_nmrdataset(fft, fftn, args))
    #     elif key == 'test':
    #         test_data = NmrDataset(get_nmrdataset(fft, fftn, args))
    #     else:
    #         raise NotImplementedError
    #
    # train_loader = DataLoader(dataset=train_data, batch_size=args.bs, num_workers=8, shuffle=True)
    # valid_loader = DataLoader(dataset=val_data, batch_size=args.bs, num_workers=4, shuffle=True)

    sup_model_path, sup_loss = net_optimise()
    print("Supervised training finished! Saved model at " + sup_model_path + ". Performance: " + str(sup_loss))

