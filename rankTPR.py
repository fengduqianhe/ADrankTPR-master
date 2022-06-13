# -*- coding: utf-8 -*-
# @Author  : qiaohezhe
# @github : https://github.com/fengduqianhe
# @Date    :  2021/3/7 
# version： Python 3.7.8
# @File : rankTPR.py
# @Software: PyCharm
import torch
from tqdm import tqdm
import torch.nn as nn
from torch import cat
import torch.nn.init as init
import math
import sys
import torch
from torchvision import datasets, models, transforms
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import pandas as pd
from adni_util import AdniDataSet
from sklearn.model_selection import train_test_split
import time
import numpy as np
import matplotlib.pyplot as plt
import os
import h5py
import torch.nn.functional as F
from datetime import datetime
from torch.optim.lr_scheduler import ReduceLROnPlateau

os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, [3]))
start = datetime.now()
seed = 0
torch.cuda.manual_seed_all(seed)


class FocalLoss(nn.Module):

    def __init__(self, weight=None, reduction='mean', gamma=0, eps=1e-7):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.eps = eps
        self.ce = torch.nn.CrossEntropyLoss(weight=weight, reduction=reduction)

    def forward(self, input, target):
        logp = self.ce(input, target)

        return loss.mean()


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(
                f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss


class FirstNet(nn.Module, ):

    def __init__(self, f=8):
        super(FirstNet, self).__init__()

        self.layer1 = nn.Sequential()
        self.layer1.add_module('conv1', nn.Conv3d(in_channels=1, out_channels=4 * f, kernel_size=3, stride=1, padding=0,
                                                  dilation=1))
        self.layer1.add_module('bn1', nn.BatchNorm3d(num_features=4 * f))
        self.layer1.add_module('relu1', nn.ReLU(inplace=True))
        self.layer1.add_module('max_pooling1', nn.MaxPool3d(kernel_size=3, stride=2))

        self.layer2 = nn.Sequential()
        self.layer2.add_module('conv2',
                               nn.Conv3d(in_channels=4 * f, out_channels=16 * f, kernel_size=3, stride=1, padding=0,
                                         dilation=2))
        self.layer2.add_module('bn2', nn.BatchNorm3d(num_features=16 * f))
        self.layer2.add_module('relu2', nn.ReLU(inplace=True))
        self.layer2.add_module('max_pooling2', nn.MaxPool3d(kernel_size=3, stride=2))

        self.layer3 = nn.Sequential()
        self.layer3.add_module('conv3',
                               nn.Conv3d(in_channels=16 * f, out_channels=32 * f, kernel_size=3, stride=1, padding=2,
                                         dilation=2))
        self.layer3.add_module('bn3', nn.BatchNorm3d(num_features=32 * f))
        self.layer3.add_module('relu3', nn.ReLU(inplace=True))
        self.layer3.add_module('max_pooling3', nn.MaxPool3d(kernel_size=3, stride=2))

        self.layer4 = nn.Sequential()
        self.layer4.add_module('conv4',
                               nn.Conv3d(in_channels=32 * f, out_channels=64 * f, kernel_size=2, stride=1, padding=1,
                                         dilation=2))
        self.layer4.add_module('bn4', nn.BatchNorm3d(num_features=64 * f))
        self.layer4.add_module('relu4', nn.ReLU(inplace=True))
        self.layer4.add_module('max_pooling4', nn.MaxPool3d(kernel_size=5, stride=2))
        
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))

        self.fc_branch_list_mmse = nn.ModuleList([nn.Linear(512, 2) for i in range(BIN_NUM_MMSE)])
        self.fc_branch_list_adas11 = nn.ModuleList([nn.Linear(512, 2) for i in range(BIN_NUM_ADAS)])
        self.fc_branch_list_adas13 = nn.ModuleList([nn.Linear(512, 2) for i in range(BIN_NUM_ADAS)])
        self.fc_branch_list_cdrsb = nn.ModuleList([nn.Linear(512, 2) for i in range(BIN_NUM_CDRSB)])

        self.fc_mmse = nn.Sequential()
        self.fc_mmse.add_module('fc1', nn.Linear(512, 256))
        self.fc_mmse.add_module('fc2', nn.Linear(256, 64))
        self.fc_mmse.add_module('fc3', nn.Linear(64, BIN_NUM_MMSE))

        self.fc_adas11 = nn.Sequential()
        self.fc_adas11.add_module('fc1', nn.Linear(512, 256))
        self.fc_adas11.add_module('fc2', nn.Linear(256, 64))
        self.fc_adas11.add_module('fc3', nn.Linear(64, BIN_NUM_ADAS))

        self.fc_adas13 = nn.Sequential()
        self.fc_adas13.add_module('fc1', nn.Linear(512, 256))
        self.fc_adas13.add_module('fc2', nn.Linear(256, 64))
        self.fc_adas13.add_module('fc3', nn.Linear(64, BIN_NUM_ADAS))

        self.fc_cdrsb = nn.Sequential()
        self.fc_cdrsb.add_module('fc1', nn.Linear(512, 256))
        self.fc_cdrsb.add_module('fc2', nn.Linear(256, 64))
        self.fc_cdrsb.add_module('fc3', nn.Linear(64, BIN_NUM_CDRSB))

    def forward(self, x1):
        x1 = self.layer1(x1)
        x1 = self.layer2(x1)
        x1 = self.layer3(x1)
        x1 = self.layer4(x1)
        x1= self.avgpool(x1)
        x1 = x1.view(x1.shape[0], -1)
        x2 = x1

        branch_mmse = []
        for i_num in range(0, BIN_NUM_MMSE):
            single_branch_mmse = self.fc_branch_list_mmse[i_num].forward(x2.unsqueeze(1))
            branch_mmse.append(single_branch_mmse)
        branch_mmse = torch.cat(branch_mmse, dim=1)

        branch_adas13 = []
        for i_num in range(0, BIN_NUM_ADAS):
            single_branch_adas13 = self.fc_branch_list_adas13[i_num].forward(x2.unsqueeze(1))
            branch_adas13.append(single_branch_adas13)
        branch_adas13 = torch.cat(branch_adas13, dim=1)

        branch_adas11 = []
        for i_num in range(0, BIN_NUM_ADAS):
            single_branch_11 = self.fc_branch_list_adas11[i_num].forward(x2.unsqueeze(1))
            branch_adas11.append(single_branch_11)
        branch_adas11 = torch.cat(branch_adas11, dim=1)

        branch_cdrsb = []
        for i_num in range(0, BIN_NUM_CDRSB):
            single_branch_cdrsb = self.fc_branch_list_cdrsb[i_num].forward(x2.unsqueeze(1))
            branch_cdrsb.append(single_branch_cdrsb)
        branch_cdrsb = torch.cat(branch_cdrsb, dim=1)

        x_mmse = self.fc_mmse(x1)
        x_adas13 = self.fc_adas13(x1)
        x_adas11 = self.fc_adas11(x1)
        x_cdrsb = self.fc_cdrsb(x1)

        return x_mmse, x_adas13, x_adas11, x_cdrsb, branch_mmse, branch_adas13, branch_adas11, branch_cdrsb


if __name__ == "__main__":
    NUM_CHNS = 1
    FEATURE_DEPTH = [32, 64, 64, 128, 128, 64]
    NUM_REGION_FEATURES = 64
    NUM_SUBJECT_FEATURES = 64

    WITH_BN = True
    WITH_DROPOUT = True
    DROP_PROB = 0.5

    SHARING_WEIGHTS = False

    PATCH_SIZE = 25

    TRN_BATCH_SIZE = 5
    TST_BATCH_SIZE = 5
    IMAGE_SIZE = 90
    NUM_EPOCHS = 100

    BIN_LARGE_MMSE = 40
    BIN_LARGE_ADAS = 80
    BIN_LARGE_CDRSB = 20

    BIN_NUM_MMSE = 8
    BIN_NUM_ADAS = 16
    BIN_NUM_CDRSB = 10


    BIN_SIZE_MMSE = 5
    BIN_SIZE_ADAS = 5
    BIN_SIZE_CDRSB = 2

    DATASET = 'all'

    # IMAGE1_PATH = "/data/MRI_MMSE/BL818_processed/"
    # IMAGE2_PATH = "/data/MRI_MMSE/BL776_processed/"
    # #
    # ADNI1_DATA = pd.read_csv("/data/ADNIMERGE_ADNI1_BL_PROCESSED.csv")
    # ADNI2_DATA = pd.read_csv("/data/ADNIMERGE_ADNI2_BL_PROCESSED.csv")
    # ADNI1_DATA = pd.read_csv("/data/ADNIMERGE_ADNI1_12_PROCESSED.csv")
    # ADNI2_DATA = pd.read_csv("/data/ADNIMERGE_ADNI2_12_PROCESSED.csv")
    # ADNI1_DATA = pd.read_csv("/data/ADNIMERGE_ADNI1_06_PROCESSED.csv")
    # ADNI2_DATA = pd.read_csv("/data/ADNIMERGE_ADNI2_06_PROCESSED.csv")
    # ADNI1_DATA = pd.read_csv("/data/ADNIMERGE_ADNI1_24_PROCESSED.csv")
    # ADNI2_DATA = pd.read_csv("/data/ADNIMERGE_ADNI2_24_PROCESSED.csv")

    IMAGE2_PATH = "/data/MRI_MMSE/BL818_processed/"
    IMAGE1_PATH = "/data/MRI_MMSE/BL776_processed/"

    ADNI2_DATA = pd.read_csv("/data/ADNIMERGE_ADNI1_BL_PROCESSED.csv")
    ADNI1_DATA = pd.read_csv("/data/ADNIMERGE_ADNI2_BL_PROCESSED.csv")
    # ADNI2_DATA = pd.read_csv("/data/ADNIMERGE_ADNI1_06_PROCESSED.csv")
    # ADNI1_DATA = pd.read_csv("/data/ADNIMERGE_ADNI2_06_PROCESSED.csv")
    # ADNI2_DATA = pd.read_csv("/data/ADNIMERGE_ADNI1_12_PROCESSED.csv")
    # ADNI1_DATA = pd.read_csv("/data/ADNIMERGE_ADNI2_12_PROCESSED.csv")
    # ADNI2_DATA = pd.read_csv("/data/ADNIMERGE_ADNI1_24_PROCESSED.csv")
    # ADNI1_DATA = pd.read_csv("/data/ADNIMERGE_ADNI2_24_PROCESSED.csv")

    TRN_LBLS_MMSE = ADNI1_DATA['MMSE'].tolist()
    VAL_LBLS_MMSE = ADNI2_DATA['MMSE'].tolist()

    TRN_LBLS_ADAS13 = ADNI1_DATA['ADAS13'].tolist()
    VAL_LBLS_ADAS13 = ADNI2_DATA['ADAS13'].tolist()

    TRN_LBLS_ADAS11 = ADNI1_DATA['ADAS11'].tolist()
    VAL_LBLS_ADAS11 = ADNI2_DATA['ADAS11'].tolist()

    TRN_LBLS_CDRSB = ADNI1_DATA['CDRSB'].tolist()
    VAL_LBLS_CDRSB = ADNI2_DATA['CDRSB'].tolist()

    TRN_SUBJECT_IDXS = ADNI1_DATA['SID'].tolist()
    VAL_SUBJECT_IDXS = ADNI2_DATA['SID'].tolist()

    print(len(TRN_SUBJECT_IDXS))
    print(len(VAL_SUBJECT_IDXS))
    TRN_STEPS = int(np.round(len(TRN_SUBJECT_IDXS) / TRN_BATCH_SIZE))
    TST_STEPS = int(np.round(len(VAL_SUBJECT_IDXS) / TST_BATCH_SIZE))

    train_subject_num = len(TRN_SUBJECT_IDXS)
    val_subject_num = len(VAL_SUBJECT_IDXS)

    train_flow = AdniDataSet(IMAGE1_PATH, TRN_SUBJECT_IDXS, TRN_LBLS_MMSE, TRN_LBLS_ADAS11, TRN_LBLS_ADAS13,
                             TRN_LBLS_CDRSB,
                             IMAGE_SIZE, BIN_SIZE_MMSE, BIN_NUM_MMSE, BIN_SIZE_ADAS, BIN_NUM_ADAS, BIN_SIZE_CDRSB,
                             BIN_NUM_CDRSB)
    test_flow = AdniDataSet(IMAGE2_PATH, VAL_SUBJECT_IDXS, VAL_LBLS_MMSE, VAL_LBLS_ADAS11, VAL_LBLS_ADAS13,
                            VAL_LBLS_CDRSB,
                            IMAGE_SIZE, BIN_SIZE_MMSE, BIN_NUM_MMSE, BIN_SIZE_ADAS, BIN_NUM_ADAS, BIN_SIZE_CDRSB,
                            BIN_NUM_CDRSB)

    train_loader = DataLoader(dataset=train_flow, batch_size=12, shuffle=True)
    val_loader = DataLoader(dataset=test_flow, batch_size=12, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FirstNet(f=8)
    # model = ResNet(BasicBlock, [3, 4, 6, 3], get_inplanes())
    model = torch.nn.DataParallel(model)

    print(model)
    criterion = nn.CrossEntropyLoss()
    bce = nn.BCEWithLogitsLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=1e-2)
    # optimizer = optim.Adam(model.parameters(), lr=0.0001)
    early_stopping = EarlyStopping(patience=70, verbose=True)
    model.to(device)
    result_list = []
    epochs = 150

    print("start training epoch {}".format(epochs))
    for epoch in range(epochs):
        print("Epoch{}:".format(epoch + 1))
        correct = 0
        total = 0
        running_loss = 0
        model.train()
        for i, data in tqdm(enumerate(train_loader), total=len(train_loader)):
            inputs, label_mmse, label_adas13, label_adas11, label_cdrsb, \
            line_mmse, line_adas13, line_adas11, line_cdrsb, \
            branch_mmse, branch_adas13, branch_adas11, branch_cdrsb = data

            inputs, label_mmse, label_adas13, label_adas11, label_cdrsb, \
            line_mmse, line_adas13, line_adas11, line_cdrsb, \
            branch_mmse, branch_adas13, branch_adas11, branch_cdrsb = inputs.to(
                device), label_mmse.to(device), label_adas13.to(device), label_adas11.to(device), label_cdrsb.to(
                device), \
                                                                                                                       line_mmse.to(
                                                                                                                           device), line_adas13.to(
                device), line_adas11.to(device), line_cdrsb.to(device), \
                                                                                                                       branch_mmse.to(
                                                                                                                           device), branch_adas13.to(
                device), branch_adas11.to(device), branch_cdrsb.to(device)

            optimizer.zero_grad()

            logps_mmse, logps_adas13, logps_adas11, logps_cdrsb, \
            out_branch_mmse, out_branch_adas13, out_branch_adas11, out_branch_cdrsb \
                = model.forward(inputs)
            print("train ground truth", line_mmse)
            print("train ground truth", branch_mmse)
            print("train predict", logps_mmse)

            print("train ground truth", line_adas13)
            print("train ground truth", branch_adas13)
            print("train predict", logps_adas13)

            loss_mmse = F.kl_div(torch.log_softmax(logps_mmse, 1), line_mmse.float())
            loss_adas13 = F.kl_div(torch.log_softmax(logps_adas13, 1), line_adas13.float())
            loss_adas11 = F.kl_div(torch.log_softmax(logps_adas11, 1), line_adas11.float())
            loss_cdrsb = F.kl_div(torch.log_softmax(logps_cdrsb, 1), line_cdrsb.float())

            pre_labels_mmse = torch.tensor([i for i in range(0, BIN_LARGE_MMSE, BIN_SIZE_MMSE)]).float().to(device)
            pre_labels_adas13 = torch.tensor([i for i in range(0, BIN_LARGE_ADAS, BIN_SIZE_ADAS)]).float().to(device)
            pre_labels_adas11 = torch.tensor([i for i in range(0, BIN_LARGE_ADAS, BIN_SIZE_ADAS)]).float().to(device)
            pre_labels_cdrsb = torch.tensor([i for i in range(0, BIN_LARGE_CDRSB, BIN_SIZE_CDRSB)]).float().to(device)

            print("train ground truth", label_mmse)
            print("train predict", torch.sum(torch.softmax(logps_mmse, 1) * pre_labels_mmse, 1))

            print("train ground truth", label_adas13)
            print("train predict", torch.sum(torch.softmax(logps_adas13, 1) * pre_labels_adas13, 1))

            '''branch_loss'''
            loss_branch_mmse = torch.tensor(0).float().to(device)
            loss_branch_adas13 = torch.tensor(0).float().to(device)
            loss_branch_adas11 = torch.tensor(0).float().to(device)
            loss_branch_cdrsb = torch.tensor(0).float().to(device)

            for i in range(BIN_NUM_MMSE):
                loss_branch_mmse += F.cross_entropy(out_branch_mmse[:, i], branch_mmse[:, i].long())
            for i in range(BIN_NUM_ADAS):
                loss_branch_adas13 += F.cross_entropy(out_branch_adas13[:, i], branch_adas13[:, i].long())
            for i in range(BIN_NUM_ADAS):
                loss_branch_adas11 += F.cross_entropy(out_branch_adas11[:, i], branch_adas11[:, i].long())
            for i in range(BIN_NUM_CDRSB):
                loss_branch_cdrsb += F.cross_entropy(out_branch_cdrsb[:, i], branch_cdrsb[:, i].long())

            print("branch", branch_mmse)
            print("loss_branch", loss_branch_mmse)

            loss = loss_mmse + loss_adas13 + loss_adas11 + loss_cdrsb +  (loss_branch_mmse
                   + loss_branch_adas13 + loss_branch_adas11 + loss_branch_cdrsb)

            print("loss_mmse", loss_mmse)
            print("loss_branch_mmse", loss_branch_mmse)

            print("loss_adas13", loss_adas13)
            print("loss_branch_adas13", loss_branch_adas13)

            print("loss_adas11", loss_adas11)
            print("loss_branch_adas11", loss_branch_adas11)

            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        train_loss = running_loss / len(train_loader)
        print('Epoch[{}/{}], train_loss:{:.4f}'.format(epoch + 1, epochs, train_loss))

        val_running_loss_mmse = 0
        val_running_loss_adas13 = 0
        val_running_loss_adas11 = 0
        val_running_loss_cdrsb = 0

        model.eval()
        cc_label_mmse = []
        cc_predict_mmse = []

        cc_label_adas13 = []
        cc_predict_adas13 = []

        cc_label_adas11 = []
        cc_predict_adas11 = []

        cc_label_cdrsb = []
        cc_predict_cdrsb = []

        with torch.no_grad():
            print("validation...")
            for data in val_loader:
                inputs, label_mmse, label_adas13, label_adas11, label_cdrsb, \
                line_mmse, line_adas13, line_adas11, line_cdrsb, \
                branch_mmse, branch_adas13, branch_adas11, branch_cdrsb = data

                inputs, label_mmse, label_adas13, label_adas11, label_cdrsb, \
                line_mmse, line_adas13, line_adas11, line_cdrsb, \
                branch_mmse, branch_adas13, branch_adas11, branch_cdrsb = inputs.to(
                    device), label_mmse.to(device), label_adas13.to(device), label_adas11.to(device), label_cdrsb.to(
                    device), \
                                                                                                                           line_mmse.to(
                                                                                                                               device), line_adas13.to(
                    device), line_adas11.to(device), line_cdrsb.to(device), \
                                                                                                                           branch_mmse.to(
                                                                                                                               device), branch_adas13.to(
                    device), branch_adas11.to(device), branch_cdrsb.to(device)

                logps_mmse, logps_adas13, logps_adas11, logps_cdrsb, \
                out_branch_mmse, out_branch_adas13, out_branch_adas11, out_branch_cdrsb \
                    = model.forward(inputs)

                pre_labels_mmse = torch.tensor([i for i in range(0, BIN_LARGE_MMSE, BIN_SIZE_MMSE)]).float().to(device)
                pre_labels_adas13 = torch.tensor([i for i in range(0, BIN_LARGE_ADAS, BIN_SIZE_ADAS)]).float().to(
                    device)
                pre_labels_adas11 = torch.tensor([i for i in range(0, BIN_LARGE_ADAS, BIN_SIZE_ADAS)]).float().to(
                    device)
                pre_labels_cdrsb = torch.tensor([i for i in range(0, BIN_LARGE_CDRSB, BIN_SIZE_CDRSB)]).float().to(
                    device)

                pre_labels_mmse = torch.sum(torch.softmax(logps_mmse, 1) * pre_labels_mmse, 1)
                pre_labels_adas13 = torch.sum(torch.softmax(logps_adas13, 1) * pre_labels_adas13, 1)
                pre_labels_adas11 = torch.sum(torch.softmax(logps_adas11, 1) * pre_labels_adas11, 1)
                pre_labels_cdrsb = torch.sum(torch.softmax(logps_cdrsb, 1) * pre_labels_cdrsb, 1)

                print("val ground truth", label_mmse)
                print("val predict", pre_labels_mmse)

                loss_mmse = F.mse_loss(label_mmse.float(), pre_labels_mmse)
                loss_adas13 = F.mse_loss(label_adas13.float(), pre_labels_adas13)
                loss_adas11 = F.mse_loss(label_adas11.float(), pre_labels_adas11)
                loss_cdrsb = F.mse_loss(label_cdrsb.float(), pre_labels_cdrsb)

                val_running_loss_mmse += loss_mmse.item()
                val_running_loss_adas13 += loss_adas13.item()
                val_running_loss_adas11 += loss_adas11.item()
                val_running_loss_cdrsb += loss_cdrsb.item()

                '''cc calculate '''
                cc_label_mmse += label_mmse.tolist()
                cc_predict_mmse += pre_labels_mmse.tolist()

                cc_label_adas13 += label_adas13.tolist()
                cc_predict_adas13 += pre_labels_adas13.tolist()

                cc_label_adas11 += label_adas11.tolist()
                cc_predict_adas11 += pre_labels_adas11.tolist()

                cc_label_cdrsb += label_cdrsb.tolist()
                cc_predict_cdrsb += pre_labels_cdrsb.tolist()

            val_loss_mmse = val_running_loss_mmse / len(val_loader)
            val_loss_adas13 = val_running_loss_adas13 / len(val_loader)
            val_loss_adas11 = val_running_loss_adas11 / len(val_loader)
            val_loss_cdrsb = val_running_loss_cdrsb / len(val_loader)

            print('Epoch[{}/{}], Loss:{:.4f}'.format(epoch + 1, epochs, val_loss_mmse))

            cc_label_mmse = pd.Series(cc_label_mmse)
            cc_predict_mmse = pd.Series(cc_predict_mmse)
            corr_mmse = cc_label_mmse.corr(cc_predict_mmse)

            cc_label_adas13 = pd.Series(cc_label_adas13)
            cc_predict_adas13 = pd.Series(cc_predict_adas13)
            corr_adas13 = cc_label_adas13.corr(cc_predict_adas13)

            cc_label_adas11 = pd.Series(cc_label_adas11)
            cc_predict_adas11 = pd.Series(cc_predict_adas11)
            corr_adas11 = cc_label_adas11.corr(cc_predict_adas11)

            cc_label_cdrsb = pd.Series(cc_label_cdrsb)
            cc_predict_cdrsb = pd.Series(cc_predict_cdrsb)
            corr_cdrsb = cc_label_cdrsb.corr(cc_predict_cdrsb)
            print("Corr", corr_mmse)

            label_predict = pd.concat([cc_label_cdrsb, cc_predict_cdrsb], 1)
            label_predict.to_csv(
                "/data/mmse_normal/log/all_corr/adni2_cdrsb_bl/correlation_{}_{}.csv".format(epoch, corr_cdrsb), mode='w',
                index=False, header=True)

            label_predict = pd.concat([cc_label_adas11, cc_predict_adas11], 1)
            label_predict.to_csv(
                "/data/mmse_normal/log/all_corr/adni2_adas11_bl/correlation_{}_{}.csv".format(epoch,
                                                                                                        corr_adas11),
                mode='w',
                index=False, header=True)

            label_predict = pd.concat([cc_label_adas13, cc_predict_adas13], 1)
            label_predict.to_csv(
                "/data/mmse_normal/log/all_corr/adni2_adas13_bl/correlation_{}_{}.csv".format(epoch,
                                                                                                        corr_adas13),
                mode='w',
                index=False, header=True)

            label_predict = pd.concat([cc_label_mmse, cc_predict_mmse], 1)
            label_predict.to_csv(
                "/mmse_normal/log/all_corr/adni2_mmse_bl/correlation_{}_{}.csv".format(epoch,
                                                                                                        corr_mmse),
                mode='w',
                index=False, header=True)

            result_list.append(
                [epoch, train_loss, val_loss_mmse, val_loss_adas13, val_loss_adas11, val_loss_cdrsb,
                 corr_mmse, corr_adas13, corr_adas11, corr_cdrsb])
            name = ['epoch', 'train_loss', 'val_loss_mmse', 'val_loss_adas13', 'val_loss_adas11', 'val_loss_cdrsb',
                    'corr_mmse', 'corr_adas13', 'corr_adas11', 'corr_cdrsb']
            result = pd.DataFrame(columns=name, data=result_list)
            # early_stopping(val_loss, model)

            result.to_csv("data/mmse_normal/log/corr/bin_rank_adni1_{}_bl_0001.csv".format(DATASET),
                          mode='w',
                          index=False, header=True)

    stop = datetime.now()
    print("Running time: ", stop - start)
