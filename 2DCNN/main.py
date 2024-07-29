# -*- coding: utf-8 -*-
"""
DEEP LEARNING FOR HYPERSPECTRAL DATA.

This script allows the user to run several deep models (and SVM baselines)
against various hyperspectral datasets. It is designed to quickly benchmark
state-of-the-art CNNs on various public hyperspectral datasets.

This code is released under the GPLv3 license for non-commercial and research
purposes only.
For commercial use, please contact the authors.
"""
# Python 2/3 compatiblity
from __future__ import print_function
from __future__ import division

# Torch
import torch
import torch.utils.data as data
from torchsummary import summary
import scipy.io as io

# Numpy, scipy, scikit-image, spectral
import numpy as np
import sklearn.svm
import sklearn.model_selection
from sklearn.neighbors import KNeighborsClassifier
#from skimage import io
# Visualization
import seaborn as sns
import visdom
import pandas as pd
from collections import Counter
import itertools
import matplotlib.pyplot as plt

import matplotlib.image as mpimg
import cv2 as cv
from superpixel import slic1
import torch.optim as optim
import os
import LDA_SLIC,superpixel_create,LDA_SLIC12
from utils import grouper, sliding_window, count_sliding_window,\
                  gaus_kernel,copy_self_concatenate
from utils import metrics, convert_to_color_, convert_from_color_,\
    display_dataset, display_predictions, explore_spectrums, plot_spectrums,\
    sample_gt, build_dataset, show_results, compute_imf_weights, get_device,pca_img,get_rgb_img,\
    bilateralFilter_rgb_img,bilateralFilter_img,setup_logger
from utils import calculate_theinput_dis,jointBLF_double1,show_results1,superpixel_filter,superpixel_jointBLF1, superpixel_jointBLF2
from datasets import get_dataset, HyperX, open_file, DATASETS_CONFIG
from models import get_model,train,test,val,save_model
from skimage.segmentation import slic,mark_boundaries
from superpixel_fusion import smooth_pixel, smooth_pixel_two, smooth_pixel_one
import argparse
import time
import logging
import LDA_SLIC
print('***************************分隔符********************')
print('now,the process is here to initialization parameter ')
print('***************************分隔符********************')
# 判断数据集是否存在
dataset_names = [v['name'] if 'name' in v.keys() else k for k, v in DATASETS_CONFIG.items()]

# Argument parser for CLI interaction
# 接收输入参数，选择的模型，训练样本数量
parser = argparse.ArgumentParser(description="Run deep learning experiments on"
                                             " various hyperspectral datasets")
parser.add_argument('--dataset', type=str, default=None, choices=dataset_names,
                    help="Dataset to use.")
parser.add_argument('--model', type=str, default=None,
                    help="Model to train. Available:\n"
                    "SVM (linear), "
                    "SVM_grid (grid search on linear, poly and RBF kernels), "
                    "baseline (fully connected NN), "
                    "hu (1D CNN), "
                    "hamida (3D CNN + 1D classifier), "
                    "lee (3D FCN), "
                    "chen (3D CNN), "
                    "li (3D CNN), "
                    "he (3D CNN), "
                    "luo (3D CNN), "
                    "sharma (2D CNN), "
                    "boulch (1D semi-supervised CNN), "
                    "liu (3D semi-supervised CNN), "
                    "mou (1D RNN)")
parser.add_argument('--folder', type=str, help="Folder where to store the "
                    "datasets (defaults to the current working directory).",
                    default="./Datasets/")
parser.add_argument('--cuda', type=int, default= 0,
                    help="Specify CUDA device (defaults to -1, which learns on CPU)")
parser.add_argument('--runs', type=int, default=1, help="Number of runs (default: 1)")

parser.add_argument('--restore', type=str, default=None,
                    help="Weights to use for initialization, e.g. a checkpoint")
parser.add_argument('--usepca', type=int, default=0,
                    help="use PCA or not,0 is not, else is use ")

# Dataset options
group_dataset = parser.add_argument_group('Dataset')
group_dataset.add_argument('--training_sample', type=float, default=10,
                    help="Percentage of samples to use for training (default: 10%%)")
group_dataset.add_argument('--sampling_mode', type=str, help="Sampling mode"
                    " (random sampling ,fixed,or disjoint, default: fixed)",
                    default='random')
group_dataset.add_argument('--train_set', type=str, default=None,
                    help="Path to the train ground truth (optional, this "
                    "supersedes the --sampling_mode option)")
group_dataset.add_argument('--test_set', type=str, default=None,
                    help="Path to the test set (optional, by default "
                    "the test_set is the entire ground truth minus the training)")

# Training options
group_train = parser.add_argument_group('Training')
group_train.add_argument('--epoch', type=int, help="Training epochs (optional, if"
                    " absent will be set by the model)")
group_train.add_argument('--patch_size', type=int,
                    help="Size of the spatial neighbourhood (optional, if "
                    "absent will be set by the model)")
group_train.add_argument('--lr', type=float,
                    help="Learning rate, set by the model if not specified.")
group_train.add_argument('--class_balancing', action='store_true',
                    help="Inverse median frequency class balancing (default = False)")
group_train.add_argument('--batch_size', type=int,default=64,
                    help="Batch size (optional, if absent will be set by the model")
group_train.add_argument('--test_stride', type=int, default=1,
                     help="Sliding window step stride during inference (default = 1)")
# Data augmentation parameters
group_da = parser.add_argument_group('Data augmentation')
group_da.add_argument('--flip_augmentation', action='store_true',
                    help="Random flips (if patch_size > 1)")
group_da.add_argument('--radiation_augmentation', action='store_true',
                    help="Random radiation noise (illumination)")
group_da.add_argument('--mixture_augmentation', action='store_true',
                    help="Random mixes between spectra")

parser.add_argument('--with_exploration', action='store_true',
                    help="See data exploration visualization")
parser.add_argument('--download', type=str, default=None, nargs='+',
                    choices=dataset_names,
                    help="Download the specified datasets and quits.")

args = parser.parse_args()
CUDA_DEVICE = get_device(args.cuda)

# % of training samples
SAMPLE_PERCENTAGE = args.training_sample
# Data augmentation
FLIP_AUGMENTATION = args.flip_augmentation
RADIATION_AUGMENTATION = args.radiation_augmentation
MIXTURE_AUGMENTATION = args.mixture_augmentation
# Dataset name
DATASET = args.dataset
# USE PCA OR NOT
pca = args.usepca
# Model name
MODEL = args.model
# Number of runs (for cross-validation)
N_RUNS = args.runs
# Spatial context size (number of neighbours in each spatial direction)
PATCH_SIZE = args.patch_size
# Add some visualization of the spectra ?
DATAVIZ = args.with_exploration
# Target folder to store/download/load the datasets
FOLDER = args.folder
# Number of epochs to run
EPOCH = args.epoch
# Sampling mode, e.g random sampling
SAMPLING_MODE = args.sampling_mode
# Pre-computed weights to restore
CHECKPOINT = args.restore
# Learning rate for the SGD
LEARNING_RATE = args.lr
# Automated class balancing
CLASS_BALANCING = args.class_balancing
# Training ground truth file
TRAIN_GT = args.train_set
# Testing ground truth file
TEST_GT = args.test_set
TEST_STRIDE = args.test_stride

train_time = 0
test_time = 0
params = 0
OA = 0
AA = 0
Kappa = 0
OA_list = []
AA_list = []
Kappa_list = []

snapshot_path = './checkpoints/'
if not os.path.isdir(snapshot_path):
    os.makedirs(snapshot_path, exist_ok=True)
    
setup_logger(logger_name="train", root=snapshot_path, screen=True, tofile=True)
logger = logging.getLogger(f"train")


if args.download is not None and len(args.download) > 0:
    for dataset in args.download:
        get_dataset(dataset, target_folder=FOLDER)
    quit()
# 命名可视化界面名称
viz = visdom.Visdom(env=DATASET + '  ' + MODEL)
if not viz.check_connection:
    print("Visdom is not connected. Did you run 'python -m visdom.server' ?")
# Parameters for the SVM grid search
SVM_GRID_PARAMS = [
    {"kernel": ["rbf"], "gamma": [1e-1, 1e-2, 1e-3], "C": [1, 10, 100, 1000]},
    {"kernel": ["linear"], "C": [0.1, 1, 10, 100, 1000]},
    {"kernel": ["poly"], "degree": [3], "gamma": [1e-1, 1e-2, 1e-3]},
]
hyperparams = vars(args)

# Load the dataset  此时输出的 img 是归一化之后的 img

img, img_original,gt, LABEL_VALUES, IGNORED_LABELS, RGB_BANDS, palette = get_dataset(DATASET,FOLDER)
# gt[gt != 6] = 0
# imgg = np.sum(img,2) / img.shape[2]
# prediction_data2 = pd.DataFrame(gt)
# data2 = pd.ExcelWriter('Salinas_gt.xlsx')  # 写入Excel文件
# prediction_data2.to_excel(data2, 'Salinas_gt', float_format='%.5f')  # ‘page_1’是写入excel的sheet名
# data2.save()
# quit()
# print(img[1, 1, :])
# print(img[1, 1, 0],img[1, 1, 1],img[1, 1, 0] - img[1, 1, 1])
# if DATASET == "houston2013":

#     img = img[3:346, 635:1186, :]
#     gt = gt[3:346, 635:1186]
#     gt[gt == 9] = 3
#     gt[gt == 12] = 9
#     gt[gt == 13] = 10

#     N_CLASSES = 10

if DATASET == "houston2018":
    img = img[:, 1600:, :]
    gt = gt[:, 1600:]
    gt[gt == 18] = 3
    gt[gt == 19] = 8

if DATASET == "WHU-Hi-HongHu":
    img = np.concatenate((img[:240, :, :], img[500:, :, :]), axis=0)
    gt = np.concatenate((gt[:240, :], gt[500:, :]), axis=0)
    print(img.shape, gt.shape)

img_pca = pca_img(img, img.shape[0], img.shape[1], img.shape[2], 3)

N_CLASSES = len(LABEL_VALUES)
print('***************************分隔符**************************')
print('N_CLASSES 的大小 : ',N_CLASSES)
# Number of bands (last dimension of the image tensor)
N_BANDS = img.shape[-1]
print('***************************分隔符**************************')
#print('输入高光谱数据的维度：',img.shape[0],img.shape[1],img.shape[2])

# Parameters for the SVM grid search

if palette is None:
    # Generate color palette
    palette = {0: (0, 0, 0)}
    for k, color in enumerate(sns.color_palette("hls", len(LABEL_VALUES) - 1)):
        palette[k + 1] = tuple(np.asarray(255 * np.array(color), dtype='uint8'))
invert_palette = {v: k for k, v in palette.items()}

def convert_to_color(x):
    return convert_to_color_(x, palette=palette)
def convert_from_color(x):
    return convert_from_color_(x, palette=invert_palette)

# Instantiate the experiment based on predefined networks
hyperparams.update({'n_classes': N_CLASSES, 'n_bands': N_BANDS, 'ignored_labels': IGNORED_LABELS, 'device': CUDA_DEVICE})
hyperparams = dict((k, v) for k, v in hyperparams.items() if v is not None)

# Show the image and the ground truth
display_dataset(img, gt, RGB_BANDS, LABEL_VALUES, palette, viz)


color_gt = convert_to_color(gt)
print(color_gt.shape)
rgb_img = get_rgb_img(img,RGB_BANDS)

if DATAVIZ:
    # Data exploration : compute and show the mean spectrums
    mean_spectrums = explore_spectrums(img, gt, LABEL_VALUES, viz,
                                       ignored_labels=IGNORED_LABELS)
    plot_spectrums(mean_spectrums, viz, title='Mean spectrum/class')

results = []

for run in range(N_RUNS):
    # Sample random training spectra
    print('***************************分隔符**************************')
    print('now,the process is here to create the train_gt and test_gt')
    print('***************************分隔符**************************')
    print(SAMPLE_PERCENTAGE,SAMPLING_MODE)
    # quit()
    if SAMPLE_PERCENTAGE > 1:
        SAMPLING_MODE = "fixed"
    train_gt, test_gt = sample_gt(gt, SAMPLE_PERCENTAGE, mode=SAMPLING_MODE)
    print('sample_gt 函数输出的 train_gt 的大小： ', train_gt.shape)
    print('sample_gt 函数输出的 test_gt 的大小： ', test_gt.shape)

    # 统计采样后的 train_gt 和 test_gt 中各样本数量
    train_gt_each_class_number = Counter(list(itertools.chain.from_iterable(train_gt)))
    print('sample_gt 函数输出的 train_gt 中各样本数量： ', train_gt_each_class_number,
          len(list(itertools.chain.from_iterable(train_gt))))
    test_gt_each_class_number = Counter(list(itertools.chain.from_iterable(test_gt)))
    print('sample_gt 函数输出的 test_gt 中各样本数量： ', test_gt_each_class_number,
          len(list(itertools.chain.from_iterable(test_gt))))
    gt_each_class_number = Counter(list(itertools.chain.from_iterable(gt)))
    print('待处理数据中  gt   中各样本数量   ： ', gt_each_class_number, len(list(itertools.chain.from_iterable(gt))))

    val_gt, val_gt1 = sample_gt(test_gt,  0.01, mode=SAMPLING_MODE)
    # 统计采样后的 train_gt 和 val_gt 中各本数量
    train_gt_each_class_number = Counter(train_gt.flatten())
    print('分验证和测试之后的，sample_gt 函数输出的 train_gt 中各样本数量： ', train_gt_each_class_number, len(train_gt.flatten()))
    train_gt_each_class_number = Counter(val_gt.flatten())
    print('分验证和测试之后的，sample_gt 函数输出的 val_gt 中各样本数量： ', train_gt_each_class_number, len(val_gt.flatten()))

    print("{} samples selected (over {})".format(np.count_nonzero(train_gt),
                                                 np.count_nonzero(gt)))
    # quit()


    display_predictions(convert_to_color(train_gt), viz, caption="Train ground truth")
    display_predictions(convert_to_color(test_gt), viz, caption="Test ground truth")
    display_predictions(convert_to_color(gt), viz, caption="Test ground truth")


    print('***************************分隔符**************************')
    print('now,the process is here to choose the model to train')
    print('***************************分隔符**************************')

    print('***************************分隔符**************************')
    print('The model is trained on a neural network ')
    print('***************************分隔符**************************')


    # ls = LDA_SLIC.LDA_SLIC(img, train_gt.astype(float), N_CLASSES - 2)
    #
    # Q1, S1, A1, Seg1 = ls.simple_superpixel(scale=120)
    #
    # out = mark_boundaries(color_gt, Seg1)
    #
    # plt.figure()
    # plt.imshow(out)
    # plt.show()
    # quit()

    if MODEL == "SVM_grid":
        print("Running a grid search SVM")
        # Grid search SVM (linear and RBF)
        X_train, y_train = build_dataset(img, train_gt, ignored_labels=IGNORED_LABELS)
        class_weight = "balanced" if CLASS_BALANCING else None
        clf = sklearn.svm.SVC(class_weight=class_weight)
        clf = sklearn.model_selection.GridSearchCV(
            clf, SVM_GRID_PARAMS, verbose=5, n_jobs=4
        )
        clf.fit(X_train, y_train)
        print("SVM best parameters : {}".format(clf.best_params_))
        prediction = clf.predict(img.reshape(-1, N_BANDS))
        save_model(clf, MODEL, DATASET)
        prediction = prediction.reshape(img.shape[:2])

        display_predictions(convert_to_color(prediction), viz, caption=" SVM prediction")
        # run_results = metrics(prediction, test_gt, hyperparams, n_classes=N_CLASSES)

        # quit()


    elif MODEL == "SVM":
        X_train, y_train = build_dataset(img, train_gt, ignored_labels=IGNORED_LABELS)
        class_weight = "balanced" if CLASS_BALANCING else None
        clf = sklearn.svm.SVC(class_weight=class_weight)
        clf.fit(X_train, y_train)
        save_model(clf, MODEL, DATASET)
        prediction = clf.predict(img.reshape(-1, N_BANDS))
        prediction = prediction.reshape(img.shape[:2])
    elif MODEL == "SGD":
        X_train, y_train = build_dataset(img, train_gt, ignored_labels=IGNORED_LABELS)
        X_train, y_train = sklearn.utils.shuffle(X_train, y_train)
        scaler = sklearn.preprocessing.StandardScaler()
        X_train = scaler.fit_transform(X_train)
        class_weight = "balanced" if CLASS_BALANCING else None
        clf = sklearn.linear_model.SGDClassifier(
            class_weight=class_weight, learning_rate="optimal", tol=1e-3, average=10
        )
        clf.fit(X_train, y_train)
        save_model(clf, MODEL, DATASET)
        prediction = clf.predict(scaler.transform(img.reshape(-1, N_BANDS)))
        prediction = prediction.reshape(img.shape[:2])
    elif MODEL == "nearest":
        X_train, y_train = build_dataset(img, train_gt, ignored_labels=IGNORED_LABELS)
        X_train, y_train = sklearn.utils.shuffle(X_train, y_train)
        class_weight = "balanced" if CLASS_BALANCING else None
        clf = KNeighborsClassifier(weights="distance")
        clf = sklearn.model_selection.GridSearchCV(
            clf, {"n_neighbors": [1, 3, 5, 10, 20]}, verbose=5, n_jobs=4
        )
        clf.fit(X_train, y_train)
        clf.fit(X_train, y_train)
        save_model(clf, MODEL, DATASET)
        prediction = clf.predict(img.reshape(-1, N_BANDS))
        prediction = prediction.reshape(img.shape[:2])
        display_predictions(convert_to_color(prediction), viz, caption=" nearest prediction")

# oss-cn-hangzhou.aliyuncs.com

    else:

        if CLASS_BALANCING:
            # if True:
            print('the process is running def compute_imf_weights')
            weights = compute_imf_weights(train_gt, N_CLASSES, IGNORED_LABELS)
            hyperparams['weights'] = torch.from_numpy(weights).cuda()

        print('when the model is trained on a neural network, the hyperparams is：', '\n', hyperparams)
        # Neural network
        model, optimizer, loss, hyperparams = get_model(MODEL, **hyperparams)
        print("Running an experiment with the {} model".format(MODEL),
              "run {}/{}".format(run + 1, N_RUNS))

        #  Generate the dataset
        patch_size = PATCH_SIZE

        # img_pad = np.pad(img, ((patch_size // 2, patch_size // 2), (patch_size // 2, patch_size // 2), (0, 0)), 'edge')  # 只能是 边缘填充
        # img_pad = np.pad(img, ((patch_size // 2, patch_size // 2), (patch_size // 2, patch_size // 2), (0, 0)))#, 'constant')
        img_pad = np.pad(img, ((patch_size // 2, patch_size // 2), (patch_size // 2, patch_size // 2), (0, 0)),
                         'symmetric')
        # gt_pad = np.pad(train_gt, ((patch_size // 2, patch_size // 2), (patch_size // 2, patch_size // 2), (0, 0)), 'edge')  # 只能是 边缘填充
        # gt_pad = np.pad(train_gt, ((patch_size // 2, patch_size // 2), (patch_size // 2, patch_size // 2), (0, 0)))#, 'constant')
        train_gt_pad = np.pad(train_gt, ((patch_size // 2, patch_size // 2), (patch_size // 2, patch_size // 2)))

        train_dataset = HyperX(img_pad, train_gt_pad, **hyperparams)  # train_dataset 包含 返回的 两个 值  return data, label

        #  使用 DataLoader 对数据集进行批处理
        #  DataLoader 函数 会 调用 HyperX_EPF 或者 HyperX 中的 def __len__(self):   可能由于 shuffle=True
        train_loader = data.DataLoader(train_dataset,
                                       batch_size=hyperparams['batch_size'],
                                       # pin_memory=hyperparams['device'],
                                       shuffle=True)
        print('***************************分隔符**************************')
        val_gt_pad = np.pad(val_gt, ((patch_size // 2, patch_size // 2), (patch_size // 2, patch_size // 2)))
        val_dataset = HyperX(img_pad, val_gt_pad, **hyperparams)

        #  DataLoader 函数 不会 调用 HyperX_EPF 或者 HyperX 中的 def __len__(self):
        val_loader = data.DataLoader(val_dataset,
                                     # pin_memory=hyperparams['device'],
                                     batch_size=hyperparams['batch_size'])
        print('***************************分隔符**************************')
        print('when the model is trained on a neural network, the hyperparams 参数是：', '\n', hyperparams)
        print('***************************分隔符**************************')
        print("when the model is trained on a neural network, the Network :")
        # #
        para = model.parameters()
        params = sum(p.numel() for p in para if p.requires_grad)
        logger.info("parameter: " + str(params))
        quit()

        with torch.no_grad():  # 不计算梯度 求导 ？
            for input, _ in train_loader:
                # print('input.size():',input.size())
                break
            summary(model.to(hyperparams['device']), input.size()[1:])
            # quit()

            # We would like to use device=hyperparams['device'] altough we have
            # to wait for torchsummary to be fixed first.
        # if CHECKPOINT is not None:
        #     model.load_state_dict(torch.load(CHECKPOINT))

        try:
            print('***************************分隔符**************************')
            print('when the model is trained on a neural network, start train model')
            print('***************************分隔符**************************')
            tr_start = time.time()
            train(MODEL, model, optimizer, loss, train_loader, hyperparams['epoch'],
                  scheduler=hyperparams['scheduler'], device=hyperparams['device'],
                  supervision=hyperparams['supervision'], val_loader=val_loader,
                  display=viz)
            tr_end = time.time()
            train_time += tr_end - tr_start

        except KeyboardInterrupt:
            # Allow the user to stop the training
            pass
        
        

        print('***************************分隔符**************************')
        print('when the model is trained on a neural network, start test model')
        print('***************************分隔符**************************')
        tic = time.time()

        probabilities = test(MODEL, model, img, test_gt, hyperparams)
        print(probabilities.shape)
        toc = time.time()
        test_time += toc - tic

        prediction = np.argmax(probabilities, axis=-1)  # 返回一个numpy数组中最大值的 索引值。当一组中同时出现几个最大值时，返回第一个最大值的索引值。
        display_predictions(convert_to_color(prediction), viz, caption=" prediction")

        name = 'probabilities.mat'
        io.savemat(name, {'probabilities': probabilities})
        name = 'prediction.mat'
        io.savemat(name, {'prediction': prediction})


    print('***************************分隔符**************************')
    print('when the model is trained on a neural network, start calculate results')
    print('***************************分隔符**************************')

    run_results = metrics(prediction, test_gt,hyperparams,n_classes=N_CLASSES)

    OA += run_results['Accuracy']
    OA_list.append(run_results['Accuracy'])
    AA += run_results["Class_Accuracy"]
    AA_list.append(run_results["Class_Accuracy"])
    Kappa += run_results['Kappa']
    Kappa_list.append(run_results["Kappa"])

    # name111 = 'prediction'
    # data111 = io.loadmat(name111)
    # prediction = data111['prediction']
    mask = np.zeros(gt.shape, dtype='bool')
    for l in IGNORED_LABELS:
        mask[gt == l] = True
    prediction[mask] = 0    

    color_prediction = convert_to_color(prediction)
    display_predictions(color_prediction, viz, caption="Prediction")

    display_predictions(color_prediction, viz,caption="Prediction")
    display_predictions(color_prediction, viz, gt=convert_to_color(gt),caption="Prediction")
    results.append(run_results)
    print('***************************分隔符**************************')
    print('显示 初始分类 结果')
    print('***************************分隔符**************************')
    # show_results(SAMPLE_PERCENTAGE,hyperparams,run_results, viz, label_values=LABEL_VALUES)
    print('***************************分隔符**************************')
    print('***************************分隔符**************************')

    if run == N_RUNS - 1:
        ERR_OA = np.std(OA_list, ddof=1)
        ERR_AA = np.std(AA_list, ddof=1)
        ERR_Kappa = np.std(Kappa_list, ddof=1)
        logger.info("Std OA is " + str(ERR_OA))
        logger.info("Std AA is " + str(ERR_AA))
        logger.info("Std Kappa is " + str(ERR_Kappa))
        logger.info("OA is " + str(OA / N_RUNS))
        logger.info("AA is {}, kappa is {}".format(AA / N_RUNS, Kappa / N_RUNS))
        logger.info("The number of training parameters is " + str(params/N_RUNS))
        logger.info("Train time is {}, test time is {}".format(train_time / N_RUNS, test_time / N_RUNS))

if N_RUNS > 1:
    print('***************************分隔符**************************')
    print('显示 n_runs 结果')
    print('***************************分隔符**************************')
    show_results(SAMPLE_PERCENTAGE,hyperparams,results, viz, label_values=LABEL_VALUES, agregated=True)
    print('***************************分隔符**************************')
    print('***************************分隔符**************************')
print(OA_list)