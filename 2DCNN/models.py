# -*- coding: utf-8 -*-
# Torch
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.optim as optim
from torch.nn import init
import math
import os
import datetime
import numpy as np
import joblib
import pandas as pd
from tqdm import tqdm
import torch.optim as optim
from activation import mish, gelu, gelu_new, swish
from attontion import PAM_Module, CAM_Module
from utils import bilateralFilter_img
from utils import grouper, sliding_window, count_sliding_window,\
                  gaus_kernel,copy_self_concatenate,sliding_window_with_test_0
from network.deform_conv import deform_conv, deform_conv_v2,deform_conv_epf,deform_conv_regular_deform

def get_model(name, **kwargs):

    """
    Instantiate and obtain a model with adequate hyperparameters

    Args:
        name: string of the model name
        kwargs: hyperparameters
    Returns:
        model: PyTorch network
        optimizer: PyTorch optimizer
        criterion: PyTorch loss Function
        kwargs: hyperparameters with sane defaults
    """
    device = kwargs.setdefault('device', torch.device('cpu'))
    n_classes = kwargs['n_classes']
    n_bands = kwargs['n_bands']
    patch_size = kwargs['patch_size']
    weights = torch.ones(n_classes)
    weights[torch.LongTensor(kwargs['ignored_labels'])] = 0.
    weights = weights.to(device)
    weights = kwargs.setdefault('weights', weights)
    print(kwargs['ignored_labels'])


    if name == 'nn':
        kwargs.setdefault('patch_size', 1)
        center_pixel = True
        model = Baseline(n_bands, n_classes,
                         kwargs.setdefault('dropout', True))
        lr = kwargs.setdefault('learning_rate', 0.0001)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss(weight=kwargs['weights'])
        kwargs.setdefault('epoch', 50)
        kwargs.setdefault('batch_size', 128)

    elif name == '2dcnn':
        kwargs.setdefault('epoch', 50)
        patch_size = kwargs.setdefault('patch_size', 8)
        center_pixel = True
        model = CNN(n_bands, n_classes)
        lr = kwargs.setdefault('learning_rate', 0.001)
        optimizer = optim.Adam(model.parameters(), lr=lr) #程序里 原来的 优化器
        # optimizer = torch.optim.Adam(model.parameters(), lr=lr)  # optimize all cnn parameters

        # optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=0.0005)
        criterion = nn.CrossEntropyLoss()
        kwargs.setdefault('batch_size', 64)
    elif name == '2dcnn_plain':
        kwargs.setdefault('epoch', 50)
        patch_size = kwargs.setdefault('patch_size', 8)
        center_pixel = True
        model = PlainNet(n_bands, patch_size,n_classes)
        lr = kwargs.setdefault('learning_rate', 0.0005)
        optimizer = optim.Adam(model.parameters(), lr=lr) #程序里 原来的 优化器
        #optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=0.0005)
        criterion = nn.CrossEntropyLoss(weight=kwargs['weights'])
        kwargs.setdefault('batch_size', 64)
    elif name == '2dcnn_deform':
        kwargs.setdefault('epoch', 50)
        patch_size = kwargs.setdefault('patch_size', 8)
        center_pixel = True
        model = DeformNet(n_bands, patch_size,n_classes)
        lr = kwargs.setdefault('learning_rate', 0.001)
        optimizer = optim.Adam(model.parameters(), lr=lr)  # 程序里 原来的 优化器
        # optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=0.0005)
        criterion = nn.CrossEntropyLoss(weight=kwargs['weights'])
        kwargs.setdefault('batch_size', 64)

    elif name == 'dbda':
        kwargs.setdefault('epoch', 100)
        patch_size = kwargs.setdefault('patch_size', 8)
        center_pixel = True
        model = DBDA_network_MISH(n_bands,n_classes)
        # model = DeformNet_EPF1(n_bands, patch_size, n_classes)
        lr = kwargs.setdefault('learning_rate', 0.001)
        optimizer = optim.Adam(model.parameters(), lr=lr, amsgrad=False)  # , weight_decay=0.0001)
        # optimizer = optim.Adam(model.parameters(), lr=lr)#, weight_decay=0.0005)
        # optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=0.0005)
        criterion = nn.CrossEntropyLoss(weight=kwargs['weights'])
        kwargs.setdefault('batch_size', 64)
    elif name == 'hybridsn':

        kwargs.setdefault('epoch', 100)
        patch_size = kwargs.setdefault('patch_size', 8)
        center_pixel = True
        model = HybridSN(n_bands,patch_size,n_classes)
        lr = kwargs.setdefault('learning_rate', 0.001)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        # optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=0.0005)
        criterion = nn.CrossEntropyLoss(weight=kwargs['weights'])
        kwargs.setdefault('batch_size', 64)

    else:
        raise KeyError("{} model is unknown.".format(name))

    model = model.to(device)
    epoch = kwargs.setdefault('epoch', 100)
    kwargs.setdefault('scheduler', optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=100//4, verbose=True))
    #kwargs.setdefault('scheduler', None)
    kwargs.setdefault('batch_size', 64)
    kwargs.setdefault('supervision', 'full')
    kwargs.setdefault('flip_augmentation', False)
    kwargs.setdefault('radiation_augmentation', False)
    kwargs.setdefault('mixture_augmentation', False)
    kwargs['center_pixel'] = center_pixel
    return model, optimizer, criterion, kwargs
#fc
class Baseline(nn.Module):
    """
    Baseline network
    """
    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Linear):
            init.kaiming_normal_(m.weight)
            init.zeros_(m.bias)

    def __init__(self, input_channels, n_classes, dropout=False):
        super(Baseline, self).__init__()
        self.use_dropout = dropout
        if dropout:
            self.dropout = nn.Dropout(p=0.5)

        self.fc1 = nn.Linear(input_channels, 2048)
        self.fc2 = nn.Linear(2048, 4096)
        self.fc3 = nn.Linear(4096, 2048)
        self.fc4 = nn.Linear(2048, n_classes)

        self.apply(self.weight_init)

    def forward(self, x):
        print('\n', 'the input data size, x.size : ', x.size())
        x = F.relu(self.fc1(x))
        print('\n', 'after F.relu(self.fc1(x)) , x.size : ', x.size())
        if self.use_dropout:
            x = self.dropout(x)
        x = F.relu(self.fc2(x))
        if self.use_dropout:
            x = self.dropout(x)
        x = F.relu(self.fc3(x))
        if self.use_dropout:
            x = self.dropout(x)
        x = self.fc4(x)
        print('\n', 'the output data size, x.size : ', x.size())
        return x
#1D-CNN
class HuEtAl(nn.Module):
    """
    Deep Convolutional Neural Networks for Hyperspectral Image Classification
    Wei Hu, Yangyu Huang, Li Wei, Fan Zhang and Hengchao Li
    Journal of Sensors, Volume 2015 (2015)
    https://www.hindawi.com/journals/js/2015/258619/

         1D-CNN

    """
    @staticmethod
    def weight_init(m):
        # [All the trainable parameters in our CNN should be initialized to
        # be a random value between −0.05 and 0.05.]
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv1d):
            init.uniform_(m.weight, -0.05, 0.05)
            init.zeros_(m.bias)

    def _get_final_flattened_size(self):
        with torch.no_grad():
            x = torch.zeros(1, 1, self.input_channels)
            x = self.pool(self.conv(x))
        return x.numel() 

    def __init__(self, input_channels, n_classes, kernel_size=None, pool_size=None):
        print('i am here,def __init__(self, input_channels, n_classes, kernel_size=None, pool_size=None)')
        super(HuEtAl, self).__init__()
        if kernel_size is None:
           # [In our experiments, k1 is better to be [ceil](n1/9)]
           kernel_size = math.ceil(input_channels / 9)
        if pool_size is None:
           # The authors recommand that k2's value is chosen so that the pooled features have 30~40 values
           # ceil(kernel_size/5) gives the same values as in the paper so let's assume it's okay
           pool_size = math.ceil(kernel_size / 5)
        self.input_channels = input_channels

        # [The first hidden convolution layer C1 filters the n1 x 1 input data with 20 kernels of size k1 x 1]
        self.conv = nn.Conv1d(1, 20, kernel_size)
        self.pool = nn.MaxPool1d(pool_size)
        self.features_size = self._get_final_flattened_size()
        # [n4 is set to be 100]
        self.fc1 = nn.Linear(self.features_size, 100)
        self.fc2 = nn.Linear(100, n_classes)
        self.apply(self.weight_init)

    def forward(self, x):
        # [In our design architecture, we choose the hyperbolic tangent function tanh(u)]
        #print('i am here, def forward(self, x)')

        # print('\n','the input data size, x.size : ',x.size())
        x = x.squeeze(dim=-1).squeeze(dim=-1)
        x = x.unsqueeze(1)
        # print('\n','after x.squeeze and x.unsqueeze, the input data size, x.size : ', x.size())

        # 先看torch.squeeze()
        # 这个函数主要对数据的维度进行压缩，去掉维数为1的的维度，比如是一行或者一列这种，一个一行三列（1, 3）的数去掉第一个维数为一的维度之后就变成（3）行。squeeze(a)
        # 就是将a中所有为1的维度删掉。不为1的维度没有影响。a.squeeze(N)
        # 就是去掉a中指定的维数为一的维度。还有一种形式就是b = torch.squeeze(a，N) a中去掉指定的定的维数为一的维度。
        #
        # 再看torch.unsqueeze()
        # 这个函数主要是对数据维度进行扩充。给指定位置加上维数为一的维度，比如原本有个三行的数据（3），在0的位置加了一维就变成一行三列（1, 3）。a.squeeze(N)
        # 就是在a中指定位置N加上一个维数为1的维度。还有一种形式就是b = torch.squeeze(a，N) a就是在a中指定位置N加上一个维数为1的维度

        x = self.conv(x)
        # print('\n','after self.conv(x), the input data size, x.size : ', x.size())
        x = torch.tanh(self.pool(x))
        x = x.view(-1, self.features_size)
        x = torch.tanh(self.fc1(x))
        x = self.fc2(x)
        # print('\n', 'after self.fc2(x), the input data size, x.size : ', x.size())
        return x

#---------------------------------------------------------------
#                 nn.Sequential
#一个有序的容器，神经网络模块将按照在传入构造器的顺序依次被添加到计算图中执行，
# 同时以神经网络模块为元素的有序字典也可以作为传入参数。
#---------------------------------------------------------------


#---------------------------------------------------------------
#               nn.BatchNorm2d
#在卷积神经网络的卷积层之后总会添加BatchNorm2d进行数据的归一化处理，
#这使得数据在进行Relu之前不会因为数据过大而导致网络性能的不稳定
#---------------------------------------------------------------

#2D-CNN  第一种

class CNN(nn.Module):
    def __init__(self,input_channels, n_classes):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=input_channels,out_channels=32,kernel_size=3,stride=1,padding=1,),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            # nn.Dropout(0.1),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveMaxPool2d(1),
            # nn.Dropout(0.1),
        )
        # self.conv4 = nn.Sequential(
        #     nn.Conv2d(128, 256, 3, 1, 1),
        #     nn.BatchNorm2d(256),
        #     nn.ReLU(),
        #     nn.AdaptiveMaxPool2d(1),
        #     #nn.Dropout(0.5),
        # )
        self.out = nn.Linear(128, n_classes)  # fully connected layer, output 16 classes

    def forward(self, x):
        # print(x.size())
        x = self.conv1(x)
        # print(x.size())
        x = self.conv2(x)
        # print(x.size())
        x = self.conv3(x)
        # print(x.size())
        # x = self.conv4(x)
        # print(x.size())
        x = x.view(x.size(0), -1)  # flatten the output of conv2 to (batch_size, 32 * 7 * 7)
        # print(x.size())
        output = self.out(x)
        # print(output.size())

        # print(output.size())
        # quit()
        return output

# 2D-CNN_PLAIN  第二种
class PlainNet(nn.Module):
    def __init__(self,input_channels, patch_size,n_classes):
        super(PlainNet, self).__init__()

        self.input_channels = input_channels
        self.patch_size = patch_size

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=input_channels,out_channels=96,kernel_size=3,stride=1,padding=1,),
            # deform_conv.DeformConv2D(input_channels, 32, kernel_size=3,stride=1,padding=1,),
            nn.BatchNorm2d(96),
            nn.ReLU(),
            # nn.MaxPool2d(kernel_size=2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(96, 96, 3, 1, 1),
            nn.BatchNorm2d(96),
            nn.ReLU(),
            nn.MaxPool2d(2),
            # nn.Dropout(0.5),
        )
        self.conv3 = nn.Sequential(
            #deform_conv.DeformConv2D(64, 128, 3, 1, 1),
            nn.Conv2d(96, 108, 3, 1, 1),
            nn.BatchNorm2d(108),
            nn.ReLU(),
            # nn.AdaptiveMaxPool2d(1),
            #nn.Dropout(0.5),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(108, 108, 3, 1, 1),
            nn.BatchNorm2d(108),
            nn.ReLU(),
            nn.MaxPool2d(2),
            # nn.Dropout(0.5),
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(108, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            # nn.MaxPool2d(2),
            # nn.Dropout(0.5),
        )
        self.conv6 = nn.Sequential(
            nn.Conv2d(128, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            # deform_conv.DeformConv2D(128, 128, 3, 1, 1),
            nn.AvgPool2d(2),
            # nn.Dropout(0.5),
        )

        self.features_size = self._get_final_flattened_size()

        self.Dropout = nn.Dropout(0.1)
        self.out = nn.Linear(self.features_size, 200)  # fully connected layer, output 16 classes
        self.out1 = nn.Linear(200, n_classes)  # fully connected layer, output 16 classes

    def _get_final_flattened_size(self):
            with torch.no_grad():
                x = torch.zeros((1, self.input_channels,
                                 self.patch_size, self.patch_size))
                x = self.conv1(x)
                x = self.conv2(x)
                x = self.conv3(x)
                x = self.conv4(x)
                x = self.conv5(x)
                x = self.conv6(x)
                _, t, c, w = x.size()
            return t * c * w

    def forward(self, x):
        # print(x.size())
        x = self.conv1(x)
        # print(x.size())
        x = self.conv2(x)
        # print(x.size())
        x = self.conv3(x)
        # print(x.size())
        x = self.conv4(x)
        # print(x.size())
        x = self.conv5(x)
        # print(x.size())
        x = self.conv6(x)
        # print(x.size())
        # quit()
        x = x.contiguous().view(x.size(0), -1)  # flatten the output of conv2 to (batch_size, 32 * 7 * 7)
        # print(x.size())
        output = self.out(x)
        # print(output.size())
        output = self.Dropout(output)
        # print(output.size())
        output = self.out1(output)
        # print(output.size())
        # quit()
        return output
# 2D-DCN_DEFORM  没有用这个网络，使用的是原作者的代码
class DeformNet(nn.Module):
    def __init__(self,input_channels, patch_size,n_classes):
        super(DeformNet, self).__init__()

        self.input_channels = input_channels
        self.patch_size = patch_size

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=input_channels,out_channels=96,kernel_size=3,stride=1,padding=1,),
            # deform_conv.DeformConv2D(input_channels, 96, kernel_size=3,stride=1,padding=1,),
            nn.BatchNorm2d(96),
            nn.ReLU(),
            # nn.MaxPool2d(kernel_size=2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(96, 96, 3, 1, 1),
            nn.BatchNorm2d(96),
            nn.ReLU(),
            nn.MaxPool2d(2),
            # nn.Dropout(0.5),
        )
        self.conv3 = nn.Sequential(
            #deform_conv.DeformConv2D(64, 128, 3, 1, 1),
            nn.Conv2d(96, 108, 3, 1, 1),
            nn.BatchNorm2d(108),
            nn.ReLU(),
            # nn.AdaptiveMaxPool2d(1),
            #nn.Dropout(0.5),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(108, 108, 3, 1, 1),
            nn.BatchNorm2d(108),
            nn.ReLU(),
            nn.MaxPool2d(2),
            # nn.Dropout(0.5),
        )
        self.conv5 = nn.Sequential(
            deform_conv.DeformConv2D(108, 128, 3, 1, 1),
            # nn.Conv2d(108, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            # deform_conv.DeformConv2D(128, 128, 3, 1, 1),
            # nn.MaxPool2d(2),
            # nn.Dropout(0.5),
        )
        self.conv6 = nn.Sequential(
            nn.Conv2d(128, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            # deform_conv.DeformConv2D(128, 128, 3, 1, 1),
            nn.AvgPool2d(2),
            # nn.Dropout(0.5),
        )

        self.features_size = self._get_final_flattened_size()

        self.Dropout = nn.Dropout(0.1)
        self.out = nn.Linear(self.features_size, 200)  # fully connected layer, output 16 classes
        self.out1 = nn.Linear(200, n_classes)  # fully connected layer, output 16 classes

    def _get_final_flattened_size(self):
            with torch.no_grad():
                x = torch.zeros((1, self.input_channels,
                                 self.patch_size, self.patch_size))
                x = self.conv1(x)
                x = self.conv2(x)
                x = self.conv3(x)
                x = self.conv4(x)
                x = self.conv5(x)
                x = self.conv6(x)
                _, t, c, w = x.size()
            return t * c * w

    def forward(self, x):
        # print(x.size())
        x = self.conv1(x)
        # print(x.size())
        x = self.conv2(x)
        # print(x.size())
        x = self.conv3(x)
        # print(x.size())
        x = self.conv4(x)
        # print(x.size())
        x = self.conv5(x)
        # print(x.size())
        x = self.conv6(x)
        # print(x.size())
        # quit()
        x = x.view(x.size(0), -1)  # flatten the output of conv2 to (batch_size, 32 * 7 * 7)
        # print(x.size())
        output = self.out(x)
        # print(output.size())
        output = self.Dropout(output)
        # print(output.size())
        output = self.out1(output)
        # print(output.size())
        # quit()
        return output

class DBDA_network_MISH(nn.Module):
    def __init__(self, band, classes):
        super(DBDA_network_MISH, self).__init__()

        # spectral branch
        self.name = 'DBDA_MISH'
        self.conv11 = nn.Conv3d(in_channels=1, out_channels=24,
                                kernel_size=(1, 1, 7), stride=(1, 1, 2))
        # Dense block
        self.batch_norm11 = nn.Sequential(
                                    nn.BatchNorm3d(24,  eps=0.001, momentum=0.1, affine=True), # 动量默认值为0.1
                                    #gelu_new()
                                    #swish()
            mish()
        )
        self.conv12 = nn.Conv3d(in_channels=24, out_channels=12, padding=(0, 0, 3),
                                kernel_size=(1, 1, 7), stride=(1, 1, 1))
        self.batch_norm12 = nn.Sequential(
                                    nn.BatchNorm3d(36, eps=0.001, momentum=0.1, affine=True),
                                    #gelu_new()
                                    #swish()
            mish()
        )
        self.conv13 = nn.Conv3d(in_channels=36, out_channels=12, padding=(0, 0, 3),
                                kernel_size=(1, 1, 7), stride=(1, 1, 1))
        self.batch_norm13 = nn.Sequential(
                                    nn.BatchNorm3d(48, eps=0.001, momentum=0.1, affine=True),
                                    #gelu_new()
                                    #swish()
            mish()
        )
        self.conv14 = nn.Conv3d(in_channels=48, out_channels=12, padding=(0, 0, 3),
                                kernel_size=(1, 1, 7), stride=(1, 1, 1))
        self.batch_norm14 = nn.Sequential(
                                    nn.BatchNorm3d(60, eps=0.001, momentum=0.1, affine=True),
                                    #gelu_new()
                                    #swish()
            mish()
        )
        kernel_3d = math.floor((band - 6) / 2)
        self.conv15 = nn.Conv3d(in_channels=60, out_channels=60,
                                kernel_size=(1, 1, kernel_3d), stride=(1, 1, 1)) # kernel size随数据变化


        # Spatial Branch
        self.conv21 = nn.Conv3d(in_channels=1, out_channels=24,
                                kernel_size=(1, 1, band), stride=(1, 1, 1))
        # Dense block
        self.batch_norm21 = nn.Sequential(
                                    nn.BatchNorm3d(24, eps=0.001, momentum=0.1, affine=True),
                                    #gelu_new()
                                    #swish()
            mish()
        )
        self.conv22 = nn.Conv3d(in_channels=24, out_channels=12, padding=(1, 1, 0),
                                kernel_size=(3, 3, 1), stride=(1, 1, 1))
        self.batch_norm22 = nn.Sequential(
                                    nn.BatchNorm3d(36, eps=0.001, momentum=0.1, affine=True),
                                    #gelu_new()
                                    #swish()
            mish()
        )
        self.conv23 = nn.Conv3d(in_channels=36, out_channels=12, padding=(1, 1, 0),
                                kernel_size=(3, 3, 1), stride=(1, 1, 1))
        self.batch_norm23 = nn.Sequential(
                                    nn.BatchNorm3d(48, eps=0.001, momentum=0.1, affine=True),
                                    #gelu_new()
                                    #swish()
            mish()
        )
        self.conv24 = nn.Conv3d(in_channels=48, out_channels=12, padding=(1, 1, 0),
                                kernel_size=(3, 3, 1), stride=(1, 1, 1))


        self.conv25 = nn.Sequential(
                                nn.Conv3d(in_channels=1, out_channels=1, padding=(1, 1, 0),
                                kernel_size=(3, 3, 2), stride=(1, 1, 1)),
                                nn.Sigmoid()
        )

        self.batch_norm_spectral = nn.Sequential(
                                    nn.BatchNorm3d(60,  eps=0.001, momentum=0.1, affine=True), # 动量默认值为0.1
                                    #gelu_new(),
                                    #swish(),
            mish(),
                                    nn.Dropout(p=0.5)
        )
        self.batch_norm_spatial = nn.Sequential(
                                    nn.BatchNorm3d(60,  eps=0.001, momentum=0.1, affine=True), # 动量默认值为0.1
                                    #gelu_new(),
                                    #swish(),
            mish(),
                                    nn.Dropout(p=0.5)
        )


        self.global_pooling = nn.AdaptiveAvgPool3d(1)
        self.full_connection = nn.Sequential(
                                #nn.Dropout(p=0.5),
                                nn.Linear(120, classes) # ,
                                # nn.Softmax()
        )

        self.attention_spectral = CAM_Module(60)
        self.attention_spatial = PAM_Module(60)

        #fc = Dense(classes, activation='softmax', name='output1',
        #           kernel_initializer=RandomNormal(mean=0.0, stddev=0.01))

    def forward(self, X):
        # spectral
        x11 = self.conv11(X)
        #print('x11', x11.shape)
        x12 = self.batch_norm11(x11)
        x12 = self.conv12(x12)
        #print('x12', x12.shape)

        x13 = torch.cat((x11, x12), dim=1)
        #print('x13', x13.shape)
        x13 = self.batch_norm12(x13)
        x13 = self.conv13(x13)
        #print('x13', x13.shape)

        x14 = torch.cat((x11, x12, x13), dim=1)
        x14 = self.batch_norm13(x14)
        x14 = self.conv14(x14)

        x15 = torch.cat((x11, x12, x13, x14), dim=1)
        # print('x15', x15.shape)

        x16 = self.batch_norm14(x15)
        x16 = self.conv15(x16)
        #print('x16', x16.shape)  # 7*7*97, 60

        #print('x16', x16.shape)
        # 光谱注意力通道
        x1 = self.attention_spectral(x16)
        x1 = torch.mul(x1, x16)


        # spatial
        #print('x', X.shape)
        x21 = self.conv21(X)
        x22 = self.batch_norm21(x21)
        x22 = self.conv22(x22)

        x23 = torch.cat((x21, x22), dim=1)
        x23 = self.batch_norm22(x23)
        x23 = self.conv23(x23)

        x24 = torch.cat((x21, x22, x23), dim=1)
        x24 = self.batch_norm23(x24)
        x24 = self.conv24(x24)

        x25 = torch.cat((x21, x22, x23, x24), dim=1)
        #print('x25', x25.shape)
        # x25 = x25.permute(0, 4, 2, 3, 1)
        #print('x25', x25.shape)

        # 空间注意力机制
        x2 = self.attention_spatial(x25)
        x2 = torch.mul(x2, x25)

        # model1
        x1 = self.batch_norm_spectral(x1)
        x1 = self.global_pooling(x1)
        x1 = x1.squeeze(-1).squeeze(-1).squeeze(-1)
        x2 = self.batch_norm_spatial(x2)
        x2= self.global_pooling(x2)
        x2 = x2.squeeze(-1).squeeze(-1).squeeze(-1)

        x_pre = torch.cat((x1, x2), dim=1)
        #print('x_pre', x_pre.shape)

        # model2
        # x1 = torch.mul(x2, x16)
        # x2 = torch.mul(x2, x25)
        # x_pre = x1 + x2
        #
        #
        # x_pre = x_pre.view(x_pre.shape[0], -1)
        output = self.full_connection(x_pre)
        # output = self.fc(x_pre)
        return output

class DeformNet_EPF1(nn.Module):
    def __init__(self,input_channels, patch_size,n_classes):
        super(DeformNet_EPF1, self).__init__()

        self.input_channels = input_channels
        self.patch_size = patch_size

        self.conv1 = nn.Sequential(
            deform_conv_epf.DeformConv2D_EPF7(input_channels, 96, kernel_size=3, stride=1, padding=1,),
            # nn.Conv2d(in_channels=input_channels,out_channels=96,kernel_size=3,stride=1,padding=1,),
            # deform_conv.DeformConv2D(input_channels, 32, kernel_size=3,stride=1,padding=1,),
            nn.BatchNorm2d(96),
            nn.ReLU(),
            # nn.MaxPool2d(kernel_size=2),
        )
        self.conv2 = nn.Sequential(
            # deform_conv_epf.DeformConv2D_EPF7(96, 96, kernel_size=3, stride=1, padding=1, ),
            nn.Conv2d(96, 96, 3, 1, 1),
            nn.BatchNorm2d(96),
            nn.ReLU(),
            nn.MaxPool2d(2),
            # nn.Dropout(0.5),
        )
        self.conv3 = nn.Sequential(
            deform_conv_epf.DeformConv2D_EPF7(96, 108, kernel_size=3, stride=1, padding=1, ),
            # nn.Conv2d(96, 108, 3, 1, 1),
            nn.BatchNorm2d(108),
            nn.ReLU(),
            # nn.AdaptiveMaxPool2d(1),
            nn.Dropout(0.1),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(108, 108, 3, 1, 1),
            nn.BatchNorm2d(108),
            nn.ReLU(),
            nn.MaxPool2d(2),
            # nn.Dropout(0.5),
        )
        self.conv5 = nn.Sequential(
            # deform_conv_epf.DeformConv2D_EPF7(108, 128, kernel_size=3, stride=1, padding=1, ),
            nn.Conv2d(108, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            # nn.MaxPool2d(2),
            # nn.Dropout(0.5),
        )
        self.conv6 = nn.Sequential(
            nn.Conv2d(128, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            # deform_conv.DeformConv2D(128, 128, 3, 1, 1),
            nn.AvgPool2d(2),
            # nn.Dropout(0.5),
        )

        # self.features_size = self._get_final_flattened_size()

        self.Dropout = nn.Dropout(0.1)
        if self.patch_size in range(8, 16):
            self.out = nn.Linear(128, 200)  # fully connected layer, output 16 classes
        elif self.patch_size in range(16, 24):
            self.out = nn.Linear(128 * 4, 200)  # fully connected layer, output 16 classes
        elif self.patch_size in range(24, 32):
            self.out = nn.Linear(128 * 9, 200)  # fully connected layer, output 16 classes
        else:
            print('please check the input number of the nn.Linear ')

        self.out1 = nn.Linear(200, n_classes)  # fully connected layer, output 16 classes

    def forward(self, x):
        # print(x.size())
        x = self.conv1(x)
        # print(x.size())
        x = self.conv2(x)
        # print(x.size())
        x = self.conv3(x)
        # print(x.size())
        x = self.conv4(x)
        # print(x.size())
        x = self.conv5(x)
        # print(x.size())
        x = self.conv6(x)
        # print(x.size())
        # quit()
        x = x.contiguous().view(x.size(0), -1)  # flatten the output of conv2 to (batch_size, 32 * 7 * 7)
        # print(x.size())
        output = self.out(x)
        # print(output.size())
        output = self.Dropout(output)
        # print(output.size())
        output = self.out1(output)
        # print(output.size())
        # quit()
        return output

# HybridSN 2020
class HybridSN(nn.Module):

    def __init__(self, input_channels, patch_size,n_classes):
        super(HybridSN, self).__init__()

        self.input_channels = input_channels
        self.patch_size = patch_size

        self.conv1 = nn.Conv3d(1, 8, (3, 3,7))
        self.conv2 = nn.Conv3d(8, 16, (3, 3,5))
        self.conv3 = nn.Conv3d(16, 32, (3, 3, 3))

        self.features_size = self._get_final_flattened_size()

        self.conv4 = nn.Conv2d(self.features_size, 64, (3,3))

        self.features_size1 = self._get_final_flattened_size1()

        self.fc1 = nn.Linear(self.features_size1, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, n_classes)
        self.Dropout = nn.Dropout(0.4)

    def _get_final_flattened_size(self):
        with torch.no_grad():
            x = torch.zeros((1, 1, self.input_channels,
                             self.patch_size, self.patch_size))
            x = self.conv1(x)
            x = self.conv2(x)
            x = self.conv3(x)
            _, t, c, w, h = x.size()
        return t * c

    def _get_final_flattened_size1(self):
        with torch.no_grad():
            x = torch.zeros((1, 1, self.input_channels,
                             self.patch_size, self.patch_size))
            x = self.conv1(x)
            x = self.conv2(x)
            x = self.conv3(x)
            x = x.view(x.shape[0], -1, x.shape[3], x.shape[4])
            x = self.conv4(x)
            _, t, c, w = x.size()

        return t * c * w

    def forward(self, x):

        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        x = x.view(x.shape[0], -1, x.shape[3], x.shape[4])

        x = F.relu(self.conv4(x))

        x = x.view(x.shape[0], -1)

        x = F.relu(self.fc1(x))
        x = self.Dropout(x)
        x = F.relu(self.fc2(x))
        x = self.Dropout(x)
        x = F.relu(self.fc3(x))

        return x


def train(MODEL,net, optimizer, criterion, data_loader, epoch, scheduler,device,supervision,val_loader,display_iter=100,display=None):
    """
    Training loop to optimize a network for several epochs and a specified loss

    Args:
        net: a PyTorch model
        optimizer: a PyTorch optimizer
        data_loader: a PyTorch dataset loader
        epoch: int specifying the number of training epochs
        criterion: a PyTorch-compatible loss function, e.g. nn.CrossEntropyLoss
        device (optional): torch device to use (defaults to CPU)
        display_iter (optional): number of iterations before refreshing the
        display (False/None to switch off).
        scheduler (optional): PyTorch scheduler
        val_loader (optional): validation dataset
        supervision (optional): 'full' or 'semi'
    """

    print('***************************分隔符**************************')
    print('when run the def <train> , the Input parameter 《net》 is :', '\n', net)
    print('***************************分隔符**************************')
    print('when run the def <train> , the Input parameter 《optimizer》 is :', '\n', optimizer)
    print('***************************分隔符**************************')
    print('when run the def <train> , the Input parameter 《loss》 is ：', criterion)
    print('***************************分隔符**************************')
    print('when run the def <train> , the Input parameter 《data_loader》 is：', '\n', data_loader)
    print('***************************分隔符**************************')
    print('when run the def <train> , the Input parameter 《epoch》is：', epoch)
    print('***************************分隔符**************************')
    print('when run the def <train> , the Input parameter 《scheduler》 is：', scheduler)
    print('***************************分隔符**************************')
    print('when run the def <train> , the Input parameter 《device》 is：', device)
    print('***************************分隔符**************************')
    print('when run the def <train> , the Input parameter 《supervision》 is：', supervision)
    print('***************************分隔符**************************')
    print('when run the def <train> , the Input parameter 《val_loader》 is：', val_loader)
    print('***************************分隔符**************************')
    print('when run the def <train> , the Input parameter 《display_iter》 is：', display_iter)
    print('***************************分隔符**************************')
    print('when run the def <train> , the Input parameter 《display》 is：', display)
    print('***************************分隔符**************************')


    if criterion is None:
        raise Exception("Missing criterion. You must specify a loss function.")

    net.to(device)
    # save_epoch = epoch // 20 if epoch > 20 else 1   # 取整除 - 返回商的整数部分（向下取整）

    losses = np.zeros(1000000)
    mean_losses = np.zeros(100000000)
    iter_ = 1
    loss_win, val_win = None, None
    val_accuracies = []
    best_acc = 0.
    for e in tqdm(range(1, epoch + 1), desc="Training the network",ncols=100):
        # print('\n','when run def train,the iteration e is :', e)
        # Set the network to training mode
        net.train()
        avg_loss = 0.

        # Run the training loop for one epoch
        '''
        enumerate() 函数用于将一个可遍历的数据对象(如列表、元组或字符串)组合为一个索引序列，同时列出数据和数据下标，一般用在 for 循环当中。
        >>>seasons = ['Spring', 'Summer', 'Fall', 'Winter']
        >>> list(enumerate(seasons))
             [(0, 'Spring'), (1, 'Summer'), (2, 'Fall'), (3, 'Winter')]
        '''

        for batch_idx, (data, target) in tqdm(enumerate(data_loader), total=len(data_loader),ncols=100):

            # Load the data into the GPU if required
            data, target = data.to(device), target.to(device)
            # print(data.size())

            optimizer.zero_grad()
            if supervision == 'full':
                # print('lr:', optimizer.state_dict()['param_groups'][0]['lr'])
                # print("lr:", scheduler.get_lr())
                # print(data.shape)
                # quit()
                output = net(data)

                loss = criterion(output, target)
            else:
                raise ValueError("supervision mode \"{}\" is unknown.".format(supervision))

            loss.backward()
            optimizer.step()

            avg_loss += loss.item()
            losses[iter_] = loss.item()
            mean_losses[iter_] = np.mean(losses[max(0, iter_ - 100):iter_ + 1])

            if display_iter and iter_ % display_iter == 0: #   %  是 取模----相除 返回余数  # and 布尔"与" - 如果 x 为 False，x and y  返回 False，否则它返回 y 的计算值。
                string = 'Train (epoch {}/{}) [{}/{} ({:.0f}%)]\tLoss: {:.6f}'
                string = string.format(
                    e, epoch, batch_idx *
                    len(data), len(data) * len(data_loader),   #   len()    返回对象（字符、列表、元组等）长度或项目个数。
                    100. * batch_idx / len(data_loader), mean_losses[iter_])
                update = None if loss_win is None else 'append'
                # loss_win = display.line(
                #     X=np.arange(iter_ - display_iter, iter_),
                #     Y=mean_losses[iter_ - display_iter:iter_],
                #     win=loss_win,
                #     update=update,
                #     opts={'title': "Training loss",
                #           'xlabel': "Iterations",
                #           'ylabel': "Loss"
                #          }
                # )
                tqdm.write(string)  #如果在终端运行时需要在循环内打印点什么，就不能直接用print()了，而要用tqdm.write(str）

                # if len(val_accuracies) > 0:
                #     val_win = display.line(Y=np.array(val_accuracies),
                #                            X=np.arange(len(val_accuracies)),
                #                            win=val_win,
                #                            opts={'title': "Validation accuracy",
                #                                  'xlabel': "Epochs",
                #                                  'ylabel': "Accuracy"
                #                                 })
            iter_ += 1
            del(data, target, loss, output)

        # Update the scheduler
        avg_loss /= len(data_loader)
        if val_loader is not None:
            val_acc = val(net, val_loader, device=device, supervision=supervision)
            val_accuracies.append(val_acc)
            metric = -val_acc
        else:
            metric = avg_loss

        if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
            '''
            以下是 isinstance() 方法的语法:
            isinstance(object, classinfo)
            参数
            object -- 实例对象。
            classinfo -- 可以是直接或间接类名、基本类型或者由它们组成的元组。
            返回值
            如果对象的类型与参数二的类型（classinfo）相同则返回 True，否则返回 False。。
            '''
            # print(metric)
            scheduler.step(metric)
            # quit()
        elif scheduler is not None:
            scheduler.step()

        # Save the weights
        print('MODEL:', MODEL, '验证精度', abs(metric),'lr:', optimizer.state_dict()['param_groups'][0]['lr'])

        if abs(metric) > best_acc:
            best_acc = abs(metric)
            print('MODEL:',MODEL,'第',e,'次保存训练的模型','验证精度',best_acc)
            print('lr:',optimizer.state_dict()['param_groups'][0]['lr'])
            # print("lr:", scheduler.get_lr())
            save_model(net, MODEL, data_loader.dataset.name)

def save_model(model, model_name, dataset_name):
    model_dir = './checkpoints/' + model_name + "/" + dataset_name + "/"
    if not os.path.isdir(model_dir):
        os.makedirs(model_dir, exist_ok=True)
    if isinstance(model, torch.nn.Module):
        filename = "best_acc_epoch"
        tqdm.write("Saving neural network weights in {}".format(filename))
        torch.save(model.state_dict(), model_dir + filename + '.pth')
    else:
        filename = str(datetime.datetime.now().strftime('%Y-%m-%d'))
        tqdm.write("Saving model params in {}".format(filename))
        joblib.dump(model, model_dir + filename + '.pkl')

def test(MODEL,net, img, test_gt,hyperparams):
    """
    Test a model on a specific image
    """
    '''
      model.eval()，不启用 BatchNormalization 和 Dropout。此时pytorch会自动把BN和DropOut固定住，不会取平均，而是用训练好的值。
      不然的话，一旦test的batch_size过小，很容易就会被BN层导致生成图片颜色失真极大；
    '''

    patch_size = hyperparams['patch_size']
    center_pixel = hyperparams['center_pixel']
    model = hyperparams['model']
    dataset_name = hyperparams['dataset']
    batch_size, device = hyperparams['batch_size'], hyperparams['device']
    n_classes = hyperparams['n_classes']

    model_dir = './checkpoints/' + MODEL + "/" + dataset_name + "/" + 'best_acc_epoch.pth'

    net.load_state_dict(torch.load(model_dir))
    net.eval()

    # img = np.pad(img, ((patch_size//2, patch_size//2), (patch_size//2, patch_size//2)), 'constant', constant_values=(0, 0))


    print('when run def test, the hyperparams is: ','\n',hyperparams,'\n',
          'when run def test, patch_size is ',patch_size,'\n',
          'when run def test, n_classes is ',n_classes)

    kwargs = {'step': hyperparams['test_stride'], 'window_size': (patch_size, patch_size)}   #  --test_stride', type=int, default=1,
    probs = np.zeros(img.shape[:2] + (n_classes,))

    if model == 'DeformNet_Regular_deform1':

        img[:,:,img.shape[2] - 1] = 3

    # img = np.pad(img, ((patch_size // 2, patch_size // 2), (patch_size // 2, patch_size // 2), (0, 0)), 'edge')  # 只能是 边缘填充
    # img = np.pad(img, ((patch_size // 2, patch_size // 2), (patch_size // 2, patch_size // 2), (0, 0)))#, 'constant')
    img = np.pad(img, ((patch_size // 2, patch_size // 2), (patch_size // 2, patch_size // 2), (0, 0)), 'symmetric')

    print("when run def test, probs 的数据类型", type(probs))      # 打印数组数据类型
    print("when run def test, probs 的数组元素总数：", probs.size)  # 打印数组尺寸，即数组元素总数
    print("when run def test, probs 的数组形状：", probs.shape)    # 打印数组形状

    iterations = count_sliding_window(img,test_gt, **kwargs) // batch_size
    # print('when run def test, the iterations is ',iterations,count_sliding_window(img, **kwargs))
    # quit()
    for batch in tqdm(grouper(batch_size, sliding_window(img, test_gt,**kwargs)),
    # for batch in tqdm(grouper(batch_size, sliding_window_with_test_0(img, test_gt, ** kwargs)),
                      total=(iterations),
                      desc="Inference on the image",
                      ncols=100
                      ):
        '''
        应该是 grouper(batch_size, sliding_window(img, **kwargs)) 中的 sliding_window(img, **kwargs) ，把 img 传进来 给到 batch 或者 data ？
        '''
        with torch.no_grad():
            if patch_size == 1:
                data = [b[0][0, 0] for b in batch]
                '''
                #[x for x in range(5)]              结果[0, 1, 2, 3, 4] 
                #l1 = [1,2,3,4] [ x*2 for x in l1]  结果[2,4,6,8]
                #print('len(data) is ',len(data))   # len(data)  就是  batch_size  # data  中的数据是 batch_size 个 array 每个array是256个数据
                #print(data)
                #quit()
                '''
                data = np.copy(data)
                '''
                #print('len(data) is ',len(data))   #   len(data)  就是  batch_size
                #quit()
                '''
                data = torch.from_numpy(data)
                '''
                #print('len(data) is ',len(data))   #   len(data)  就是  batch_size
                #quit()
                '''
            else:
                data = [b[0] for b in batch]
                # print('len(data) is ',len(data))   #   len(data)  就是  batch_size

                data = np.copy(data)
                # print('len(data) is ',len(data),data.shape)   #   len(data)  就是  batch_size
                # quit()
                if model in ["dbda"]:
                    pass
                else:
                    data = data.transpose(0, 3, 1, 2)
                    # print('len(data) is ', len(data), data.shape)  # len(data)  就是  batch_size

                data = torch.from_numpy(data)
                data = data.float()
                #print('data.size() is ', data.size())  # len(data)  就是  batch_size
                if model in ["DeformNet_Regular_deform","deepnrd","2dcnn", "2dcnn_plain",
                             "2dcnn_deform", "2dcnn_deform_v2" , "2dcnn_deform_epf","dbda1"]:
                    # print('the input model is 2dcnn')
                    pass
                else:
                    data = data.unsqueeze(1)
                # print('data.size() is ',data.size())   #   len(data)  就是  batch_size
                #quit()
            indices = [b[1:] for b in batch]   #  b[1:] 从第二个元素开始截取列表
            '''
            #print('len(indices) is ', len(indices))   #,'\n','indices is ','\n',indices)  # len(data)  就是  batch_size
                                                                                     # indices is 从(0, 0, 1, 1)到 (0, 127, 1, 1)
            quit()
            '''

            data = data.to(device)
            # print('when run def test，输入到net的data形状，data.size():', data.type(), data.size())
            # quit()
            output = net(data)
            # print('when run def test，net输出的output形状，output.size():', output.size())
            #quit()
            '''
            可能是默认 load 最近的模型  此时的 data  就是 之前归一化的 img
            此时的 output 就是 每个像素 分类 各个类别的 概率
            print('output.shape is ',output.shape)          
            output.shape is  batch_size * n_class
            '''

            if isinstance(output, tuple):
                # print('when run def test, output type is not tuple')
                output = output[0]
            output = output.to('cpu')

            if patch_size == 1 or center_pixel:
                output = output.numpy()
                # print('when run def test，net输出的output形状，output.shape:', output.shape)
            else:
                output = np.transpose(output.numpy(), (0, 2, 3, 1))
                # print('when run def test，net输出的output形状，output.shape:', output.shape)
            #quit()

            for (x, y, w, h), out in zip(indices, output):
                                                              #  zip(*iterables) --> A zip object yielding tuples until an input is exhausted.
                                                              #
                                                              #     >>> list(zip('abcdefg', range(3), range(4)))
                                                              #     [('a', 0, 0), ('b', 1, 1), ('c', 2, 2)]
                                                              #
                                                              #  The zip object yields n-length tuples, where n is the number of iterables
                                                              #  passed as positional arguments to zip().  The i-th element in every tuple
                                                              #  comes from the i-th iterable argument to zip().  This continues until the
                                                              #  shortest argument is exhausted.

                #print('x, y, w, h,out is :',x, y, w, h,out)  # out 就是 1 * n_class
                                                              # xy,从 00 到 02 03 0m (m= img 的列数 - batch_size - 1)
                                                              #    从 21 到 22 23 2m
                                                              #    从 n1 到 n2 n3 nm (n= img 的行数 - batch_size - 1)
                                                              # w, h 就是 patch_size, patch_size
                #quit()
                if center_pixel:
                    probs[x, y] += out      # c += a 等效于 c = c + a
                                                              #  // 取整除 - 返回商的整数部分（向下取整）
                else:
                    probs[x:x + w, y:y + h] += out

    return probs

def val(net, data_loader, device, supervision):
# TODO : fix me using metrics()
    accuracy, total = 0., 0.
    ignored_labels = data_loader.dataset.ignored_labels
    for batch_idx, (data, target) in enumerate(data_loader):
        with torch.no_grad():
            # Load the data into the GPU if required
            data, target = data.to(device), target.to(device)
            if supervision == 'full':
                output = net(data)
            elif supervision == 'semi':
                outs = net(data)
                output, rec = outs
            _, output = torch.max(output, dim=1)
            # print(output)

            for out, pred in zip(output.view(-1), target.view(-1)):
                if out.item() in ignored_labels:
                    # print(out.item(),ignored_labels)
                    # print('hhhhh')
                    continue

                else:
                    accuracy += out.item() == pred.item()
                    total += 1
            # print(total)
    return accuracy / total
