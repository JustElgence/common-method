# -*- coding: utf-8 -*-
# @Auther   : Mingsong Li (lms-07)
# @Time     : 2023-Apr
# @Address  : Time Lab @ SDU
# @FileName : data_load_operate.py
# @Project  : AMS-M2ESL (HSIC), IEEE TGRS

import os
import math
import torch
import numpy as np
import spectral as spy
import scipy.io as sio
import torch.utils.data as Data
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn import preprocessing

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def load_data(data_set_name, data_path):
    if data_set_name == 'IP':
        data = sio.loadmat(os.path.join('IndianPines', 'Indian_pines_corrected.mat'))['indian_pines_corrected']
        labels = sio.loadmat(os.path.join('IndianPines', 'Indian_pines_gt.mat'))['indian_pines_gt']
    elif data_set_name == 'PaviaU':
        data = sio.loadmat(os.path.join('PaviaU', 'PaviaU.mat'))['paviaU']
        labels = sio.loadmat(os.path.join('PaviaU', 'PaviaU_gt.mat'))['paviaU_gt']
    elif data_set_name == 'Salinas':
        data = sio.loadmat(os.path.join('Salinas', 'Salinas_corrected.mat'))['salinas_corrected']
        labels = sio.loadmat(os.path.join('Salinas', 'Salinas_gt.mat'))['salinas_gt']
    elif data_set_name == 'PaviaC':
        img = sio.loadmat(os.path.join('PaviaC', 'Pavia.mat'))
        data = img['pavia']
        labels = sio.loadmat(os.path.join('PaviaC', 'Pavia_gt.mat'))['pavia_gt']
    elif data_set_name == 'WHU-Hi-LongKou':
        img = sio.loadmat(os.path.join('WHU-Hi-LongKou', 'WHU_Hi_LongKou.mat'))
        data = img['WHU_Hi_LongKou']
        labels = sio.loadmat(os.path.join('WHU-Hi-LongKou', 'WHU_Hi_LongKou_gt.mat'))['WHU_Hi_LongKou_gt']
    elif data_set_name == 'KSC':
        img = img = sio.loadmat(os.path.join('KSC', 'KSC.mat'))
        data = img['KSC']
        rgb_bands = (43, 21, 11)  # AVIRIS sensor
        labels = sio.loadmat(os.path.join('KSC', 'KSC_gt.mat'))['KSC_gt']
    elif data_set_name == 'Botswana':
        img = sio.loadmat(os.path.join('Botswana', 'Botswana.mat'))
        data = img['Botswana']
        labels = sio.loadmat(os.path.join('Botswana', 'Botswana_gt.mat'))['Botswana_gt']

    return data, labels


def load_HU_data(data_path):
    data = sio.loadmat(os.path.join(data_path, 'HU13_tif', "Houston13_data.mat"))['Houston13_data']
    labels_train = sio.loadmat(os.path.join(data_path, 'HU13_tif', "Houston13_gt_train.mat"))['Houston13_gt_train']
    labels_test = sio.loadmat(os.path.join(data_path, 'HU13_tif', "Houston13_gt_test.mat"))['Houston13_gt_test']

    return data, labels_train, labels_test


def standardization(data):
    height, width, bands = data.shape
    data = np.reshape(data, [height * width, bands])
    # data=preprocessing.scale(data) #
    # data = preprocessing.MinMaxScaler().fit_transform(data)
    data = preprocessing.StandardScaler().fit_transform(data)  #

    data = np.reshape(data, [height, width, bands])
    return data


def sampling(ratio_list, num_list, gt_reshape, class_count, Flag):
    all_label_index_dict, train_label_index_dict, test_label_index_dict = {}, {}, {}
    all_label_index_list, train_label_index_list, test_label_index_list = [], [], [],

    for cls in range(class_count):  # [0-15]
        cls_index = np.where(gt_reshape == cls + 1)[0]
        all_label_index_dict[cls] = list(cls_index)

        np.random.shuffle(cls_index)

        if Flag == 0:  # Fixed proportion for each category
            train_index_flag = max(int(ratio_list[0] * len(cls_index)), 1)  # at least 3 samples per class]
        # Split by num per class
        elif Flag == 1:  # Fixed quantity per category
            if len(cls_index) > num_list[0]:
                train_index_flag = num_list[0]
            else:
                train_index_flag = 15

        train_label_index_dict[cls] = list(cls_index[:train_index_flag])
        test_label_index_dict[cls] = list(cls_index[train_index_flag:])

        train_label_index_list += train_label_index_dict[cls]
        test_label_index_list += test_label_index_dict[cls]
        all_label_index_list += all_label_index_dict[cls]

    return train_label_index_list, test_label_index_list, all_label_index_list


def sampling_disjoint(gt_train_re, gt_test_re, class_count):
    all_label_index_dict, train_label_index_dict, test_label_index_dict = {}, {}, {}
    all_label_index_list, train_label_index_list, test_label_index_list = [], [], []

    for cls in range(class_count):
        cls_index_train = np.where(gt_train_re == cls + 1)[0]
        cls_index_test = np.where(gt_test_re == cls + 1)[0]

        train_label_index_dict[cls] = list(cls_index_train)
        test_label_index_dict[cls] = list(cls_index_test)

        train_label_index_list += train_label_index_dict[cls]
        test_label_index_list += test_label_index_dict[cls]
        all_label_index_list += (train_label_index_dict[cls] + test_label_index_dict[cls])

    return train_label_index_list, test_label_index_list, all_label_index_list


def applyPCA(X, numComponents=75):
    newX = np.reshape(X, (-1, X.shape[2]))
    pca = PCA(n_components=numComponents, whiten=True)
    newX = pca.fit_transform(newX)
    newX = np.reshape(newX, (X.shape[0], X.shape[1], numComponents))
    return newX


def HSI_MNF(X, MNF_ratio):
    denoised_bands = math.ceil(MNF_ratio * X.shape[-1])
    mnfr = spy.mnf(spy.calc_stats(X), spy.noise_from_diffs(X))
    denoised_data = mnfr.reduce(X, num=denoised_bands)

    return denoised_data


def data_pad_zero(data, patch_length):
    data_padded = np.lib.pad(data, ((patch_length, patch_length), (patch_length, patch_length), (0, 0)), 'constant',
                             constant_values=0)
    return data_padded

def img_show(x):
    spy.imshow(x)
    plt.show()


def index_assignment(index, row, col, pad_length):
    new_assign = {}  # dictionary.
    for counter, value in enumerate(index):
        assign_0 = value // col + pad_length
        assign_1 = value % col + pad_length
        new_assign[counter] = [assign_0, assign_1]
    return new_assign


def select_patch(data_padded, pos_x, pos_y, patch_length):
    selected_patch = data_padded[pos_x - patch_length:pos_x + patch_length + 1,
                     pos_y - patch_length:pos_y + patch_length + 1]
    return selected_patch


def select_vector(data_padded, pos_x, pos_y):
    select_vector = data_padded[pos_x, pos_y]
    return select_vector


def HSI_create_pathes(data_padded, hsi_h, hsi_w, data_indexes, patch_length, flag):
    h_p, w_p, c = data_padded.shape

    data_size = len(data_indexes)
    patch_size = patch_length * 2 + 1

    data_assign = index_assignment(data_indexes, hsi_h, hsi_w, patch_length)
    if flag == 1:
        # for spatial net data, HSI patch
        unit_data = np.zeros((data_size, patch_size, patch_size, c))
        unit_data_torch = torch.from_numpy(unit_data).type(torch.FloatTensor).to(device)
        for i in range(len(data_assign)):
            unit_data_torch[i] = select_patch(data_padded, data_assign[i][0], data_assign[i][1], patch_length)

    if flag == 2:
        # for spectral net data, HSI vector
        unit_data = np.zeros((data_size, c))
        unit_data_torch = torch.from_numpy(unit_data).type(torch.FloatTensor).to(device)
        for i in range(len(data_assign)):
            unit_data_torch[i] = select_vector(data_padded, data_assign[i][0], data_assign[i][1])

    return unit_data_torch


def generate_data_set(data_reshape, label, index):
    train_data_index, test_data_index, all_data_index = index
    x_train_set = data_reshape[train_data_index]
    y_train_set = label[train_data_index] - 1

    x_test_set = data_reshape[test_data_index]
    y_test_set = label[test_data_index] - 1

    x_all_set = data_reshape[all_data_index]
    y_all_set = label[all_data_index] - 1

    return x_train_set, y_train_set, x_test_set, y_test_set, x_all_set, y_all_set


def generate_data_set_disjoint(data_reshape, label_train, label_test, index):
    train_data_index, test_data_index, all_data_index = index
    x_train_set = data_reshape[train_data_index]
    y_train_set = label_train[train_data_index] - 1

    x_test_set = data_reshape[test_data_index]
    y_test_set = label_test[test_data_index] - 1

    # x_all_set = data_reshape[all_data_index]
    # y_all_set = label[all_data_index] - 1

    return x_train_set, y_train_set, x_test_set, y_test_set


# generating HSI patches using GPU directly.
def generate_iter(data_padded, hsi_h, hsi_w, label_reshape, index, patch_length, batch_size,
                  model_type_flag,
                  model_3D_spa_flag, last_batch_flag):
    # flag for single spatial net or single spectral net or spectral-spatial net
    data_padded_torch = torch.from_numpy(data_padded).type(torch.FloatTensor).to(device)

    # for data label
    
    train_labels = label_reshape[index[0]] - 1
    test_labels = label_reshape[index[1]] - 1

    y_tensor_train = torch.from_numpy(train_labels).type(torch.FloatTensor)
    y_tensor_test = torch.from_numpy(test_labels).type(torch.FloatTensor)
    index1 = torch.from_numpy(np.array(index[1])).type(torch.FloatTensor)
    # for data
    if model_type_flag == 1:  # data for single spatial net
        spa_train_samples = HSI_create_pathes(data_padded_torch, hsi_h, hsi_w, index[0], patch_length, 1)
        spa_test_samples = HSI_create_pathes(data_padded_torch, hsi_h, hsi_w, index[1], patch_length, 1)

        if model_3D_spa_flag == 1:  # spatial 3D patch
            spa_train_samples = spa_train_samples.unsqueeze(1)
            spa_test_samples = spa_test_samples.unsqueeze(1)

        torch_dataset_train = Data.TensorDataset(spa_train_samples, y_tensor_train)
        torch_dataset_test = Data.TensorDataset(spa_test_samples, y_tensor_test, index1)
        # torch_dataset_test = Data.TensorDataset(spa_test_samples, y_tensor_test)


    elif model_type_flag == 2:  # data for single spectral net
        spe_train_samples = HSI_create_pathes(data_padded_torch, hsi_h, hsi_w, index[0], patch_length, 2)
        spe_test_samples = HSI_create_pathes(data_padded_torch, hsi_h, hsi_w, index[1], patch_length, 2)

        torch_dataset_train = Data.TensorDataset(spe_train_samples, y_tensor_train)
        torch_dataset_test = Data.TensorDataset(spe_test_samples, y_tensor_test)

    elif model_type_flag == 3:  # data for spectral-spatial net
        # spatail data
        spa_train_samples = HSI_create_pathes(data_padded_torch, hsi_h, hsi_w, index[0], patch_length, 1)
        spa_test_samples = HSI_create_pathes(data_padded_torch, hsi_h, hsi_w, index[1], patch_length, 1)

        # spectral data
        spe_train_samples = HSI_create_pathes(data_padded_torch, hsi_h, hsi_w, index[0], patch_length, 2)
        spe_test_samples = HSI_create_pathes(data_padded_torch, hsi_h, hsi_w, index[1], patch_length, 2)

        torch_dataset_train = Data.TensorDataset(spa_train_samples, spe_train_samples, y_tensor_train)
        torch_dataset_test = Data.TensorDataset(spa_test_samples, spe_test_samples, y_tensor_test)
    last_batch_flag = 0
    if last_batch_flag == 0:
        train_iter = Data.DataLoader(dataset=torch_dataset_train, batch_size=batch_size, shuffle=True, num_workers=0)
        test_iter = Data.DataLoader(dataset=torch_dataset_test, batch_size=batch_size, shuffle=False, num_workers=0)
    elif last_batch_flag == 1:
        train_iter = Data.DataLoader(dataset=torch_dataset_train, batch_size=batch_size, shuffle=True, num_workers=0,
                                     drop_last=True)
        test_iter = Data.DataLoader(dataset=torch_dataset_test, batch_size=batch_size, shuffle=False, num_workers=0,
                                    drop_last=True)
    # train_iter = Data.DataLoader(dataset=torch_dataset_train, batch_size=batch_size, shuffle=True, num_workers=0)
    # test_iter = Data.DataLoader(dataset=torch_dataset_test, batch_size=batch_size, shuffle=False, num_workers=0)

    return train_iter, test_iter


def generate_iter_disjoint(data_padded, hsi_h, hsi_w, gt_train_re, gt_test_re, index, patch_length, batch_size,
                           model_type_flag, model_3D_spa_flag):
    data_padded_torch = torch.from_numpy(data_padded).type(torch.FloatTensor).to(device)

    train_labels = gt_train_re[index[0]] - 1
    test_labels = gt_test_re[index[1]] - 1

    y_tensor_train = torch.from_numpy(train_labels).type(torch.FloatTensor)
    y_tensor_test = torch.from_numpy(test_labels).type(torch.FloatTensor)

    # for data
    if model_type_flag == 1:  # data for single spatial net
        spa_train_samples = HSI_create_pathes(data_padded_torch, hsi_h, hsi_w, index[0], patch_length, 1)
        spa_test_samples = HSI_create_pathes(data_padded_torch, hsi_h, hsi_w, index[1], patch_length, 1)

        if model_3D_spa_flag == 1:  # spatial 3D patch
            spa_train_samples = spa_train_samples.unsqueeze(1)
            spa_test_samples = spa_test_samples.unsqueeze(1)

        torch_dataset_train = Data.TensorDataset(spa_train_samples, y_tensor_train)
        torch_dataset_test = Data.TensorDataset(spa_test_samples, y_tensor_test)

    elif model_type_flag == 2:  # data for single spectral net
        spe_train_samples = HSI_create_pathes(data_padded_torch, hsi_h, hsi_w, index[0], patch_length, 2)
        spe_test_samples = HSI_create_pathes(data_padded_torch, hsi_h, hsi_w, index[1], patch_length, 2)

        torch_dataset_train = Data.TensorDataset(spe_train_samples, y_tensor_train)
        torch_dataset_test = Data.TensorDataset(spe_test_samples, y_tensor_test)

    elif model_type_flag == 3:  # data for spectral-spatial net
        # spatail data
        spa_train_samples = HSI_create_pathes(data_padded_torch, hsi_h, hsi_w, index[0], patch_length, 1)
        spa_test_samples = HSI_create_pathes(data_padded_torch, hsi_h, hsi_w, index[1], patch_length, 1)

        # spectral data
        spe_train_samples = HSI_create_pathes(data_padded_torch, hsi_h, hsi_w, index[0], patch_length, 2)
        spe_test_samples = HSI_create_pathes(data_padded_torch, hsi_h, hsi_w, index[1], patch_length, 2)

        torch_dataset_train = Data.TensorDataset(spa_train_samples, spe_train_samples, y_tensor_train)
        torch_dataset_test = Data.TensorDataset(spa_test_samples, spe_test_samples, y_tensor_test)

    train_iter = Data.DataLoader(dataset=torch_dataset_train, batch_size=batch_size, shuffle=True, num_workers=0)
    test_iter = Data.DataLoader(dataset=torch_dataset_test, batch_size=batch_size, shuffle=False, num_workers=0)

    return train_iter, test_iter


# all) generating HSI patches for the visualization of all the labeled samples of the data set
# total) generating HSI patches for the visualization of total the samples of the data set
# in addition, all) and total) both use GPU directly
def generate_iter_total(data_padded, hsi_h, hsi_w, label_reshape, index, patch_length, batch_size, model_type_flag,
                        model_3D_spa_flag):
    data_padded_torch = torch.from_numpy(data_padded).type(torch.FloatTensor).to(device)

    if len(index) < label_reshape.shape[0]:
        total_labels = label_reshape[index] - 1
    else:
        total_labels = np.zeros(label_reshape.shape)

    y_tensor_total = torch.from_numpy(total_labels).type(torch.FloatTensor)

    if model_type_flag == 1:
        total_samples = HSI_create_pathes(data_padded_torch, hsi_h, hsi_w, index, patch_length, 1)
        if model_3D_spa_flag == 1:  # spatial 3D patch
            total_samples = total_samples.unsqueeze(1)
        torch_dataset_total = Data.TensorDataset(total_samples, y_tensor_total)

    elif model_type_flag == 2:
        total_samples = HSI_create_pathes(data_padded_torch, hsi_h, hsi_w, index, patch_length, 2)
        torch_dataset_total = Data.TensorDataset(total_samples, y_tensor_total)
    elif model_type_flag == 3:
        spa_total_samples = HSI_create_pathes(data_padded_torch, hsi_h, hsi_w, index, patch_length, 1)
        spe_total_samples = HSI_create_pathes(data_padded_torch, hsi_h, hsi_w, index, patch_length, 2)
        torch_dataset_total = Data.TensorDataset(spa_total_samples, spe_total_samples, y_tensor_total)

    total_iter = Data.DataLoader(dataset=torch_dataset_total, batch_size=batch_size, shuffle=False, num_workers=0)

    return total_iter
