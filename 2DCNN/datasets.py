# -*- coding: utf-8 -*-
"""
This file contains the PyTorch dataset for hyperspectral images and
related helpers.
"""
from collections import Counter
from utils import display_dataset,bilateralFilter_img
import spectral
import numpy as np
import torch
import torch.utils
import torch.utils.data
import itertools
from tqdm import tqdm
from custom_datasets import CUSTOM_DATASETS_CONFIG
from scipy import io, misc
import h5py
import sys
import pandas as pd
from utils import calculate_theinput_dis,bilateral_filter1
import random
import matplotlib.pyplot as plt
try:
    # Python 3
    from urllib.request import urlretrieve
except ImportError:
    # Python 2
    from urllib import urlretrieve

from utils import open_file

DATASETS_CONFIG = {
    'PaviaC': {
        # 'urls': ['http://www.ehu.eus/ccwintco/uploads/e/e3/Pavia.mat',
        #         'http://www.ehu.eus/ccwintco/uploads/5/53/Pavia_gt.mat'],
        'img': 'Pavia.mat',
        'gt': 'Pavia_gt.mat',
    },
    'Salinas': {
        # 'urls': ['http://www.ehu.eus/ccwintco/uploads/a/a3/Salinas_corrected.mat',
        #          'http://www.ehu.eus/ccwintco/uploads/f/fa/Salinas_gt.mat'],
        'img': 'Salinas_corrected.mat',
        'gt': 'Salinas_gt.mat',
    },

    'MUUFLGulfport': {
        # 'urls': ['http://www.ehu.es/ccwintco/uploads/7/72/Botswana.mat',
        #         'http://www.ehu.es/ccwintco/uploads/5/58/Botswana_gt.mat'],
        "img": "WHU_Hi_LongKou.mat",
        "gt": "WHU_Hi_LongKou_gt.mat",
    },
    'PaviaU': {
        # 'urls': ['http://www.ehu.eus/ccwintco/uploads/e/ee/PaviaU.mat',
        #          'http://www.ehu.eus/ccwintco/uploads/5/50/PaviaU_gt.mat'],
        'img': 'PaviaU.mat',
        'gt': 'PaviaU_gt.mat',
    },
    'ksc': {
        # 'urls': ['http://www.ehu.es/ccwintco/uploads/2/26/KSC.mat',
        #          'http://www.ehu.es/ccwintco/uploads/a/a6/KSC_gt.mat'],
        'img': 'ksc.mat',
        'gt': 'ksc_gt.mat',
    },
    'IndianPines': {
        # 'urls': ['http://www.ehu.eus/ccwintco/uploads/6/67/Indian_pines_corrected.mat',
        #         'http://www.ehu.eus/ccwintco/uploads/c/c4/Indian_pines_gt.mat'],
        'img': 'Indian_pines_corrected.mat',
        'gt': 'Indian_pines_gt.mat'
    },
    'Botswana': {
        # 'urls': ['http://www.ehu.es/ccwintco/uploads/7/72/Botswana.mat',
        #         'http://www.ehu.es/ccwintco/uploads/5/58/Botswana_gt.mat'],
        'img': 'Botswana.mat',
        'gt': 'Botswana_gt.mat',
    },
    'xiongan_one': {
        # 'urls': ['http://www.ehu.es/ccwintco/uploads/7/72/Botswana.mat',
        #         'http://www.ehu.es/ccwintco/uploads/5/58/Botswana_gt.mat'],
        'img': 'xiongan_one.mat',
        'gt': 'xiongan_one_gt.mat',
    },
    'xiongan_two': {
        # 'urls': ['http://www.ehu.es/ccwintco/uploads/7/72/Botswana.mat',
        #         'http://www.ehu.es/ccwintco/uploads/5/58/Botswana_gt.mat'],
        'img': 'xiongan_two.mat',
        'gt': 'xiongan_two_gt.mat',
    },
    'xiongan_four': {
        # 'urls': ['http://www.ehu.es/ccwintco/uploads/7/72/Botswana.mat',
        #         'http://www.ehu.es/ccwintco/uploads/5/58/Botswana_gt.mat'],
        'img': 'xiongan_four.mat',
        'gt': 'xiongan_four_gt.mat',
    },
    'xiongan_three': {
        # 'urls': ['http://www.ehu.es/ccwintco/uploads/7/72/Botswana.mat',
        #         'http://www.ehu.es/ccwintco/uploads/5/58/Botswana_gt.mat'],
        'img': 'xiongan_three.mat',
        'gt': 'xiongan_three_gt.mat',
    },
    'houston2013': {
        # 'urls': ['http://www.ehu.es/ccwintco/uploads/7/72/Botswana.mat',
        #         'http://www.ehu.es/ccwintco/uploads/5/58/Botswana_gt.mat'],
        'img': 'houston.mat',
        'gt': 'houston.mat',
    },
    'houston2018': {
        # 'urls': ['http://www.ehu.es/ccwintco/uploads/7/72/Botswana.mat',
        #         'http://www.ehu.es/ccwintco/uploads/5/58/Botswana_gt.mat'],
            "img": "2018_IEEE_GRSS_DFC_HSI_TR.HDR",
        "gt": "2018_IEEE_GRSS_DFC_GT_TR.tif",
    },
    'WHU-Hi-HanChuan': {
        # 'urls': ['http://www.ehu.es/ccwintco/uploads/7/72/Botswana.mat',
        #         'http://www.ehu.es/ccwintco/uploads/5/58/Botswana_gt.mat'],
        "img": "WHU_Hi_HanChuan.mat",
        "gt": "WHU_Hi_HanChuan_gt.mat",
    },
    'WHU-Hi-HongHu': {
        # 'urls': ['http://www.ehu.es/ccwintco/uploads/7/72/Botswana.mat',
        #         'http://www.ehu.es/ccwintco/uploads/5/58/Botswana_gt.mat'],
        "img": "WHU_Hi_HongHu.mat",
        "gt": "WHU_Hi_HongHu_gt.mat",
    },
    'WHU-Hi-LongKou': {
        # 'urls': ['http://www.ehu.es/ccwintco/uploads/7/72/Botswana.mat',
        #         'http://www.ehu.es/ccwintco/uploads/5/58/Botswana_gt.mat'],
        "img": "WHU_Hi_LongKou.mat",
        "gt": "WHU_Hi_LongKou_gt.mat",
    },
    'sd': {
        # 'urls': ['http://www.ehu.es/ccwintco/uploads/7/72/Botswana.mat',
        #         'http://www.ehu.es/ccwintco/uploads/5/58/Botswana_gt.mat'],
        'img': 'houston.mat',
        'gt': 'houston.mat',
   }
}

try:
    from custom_datasets import CUSTOM_DATASETS_CONFIG
    DATASETS_CONFIG.update(CUSTOM_DATASETS_CONFIG)
except ImportError:
    pass


class TqdmUpTo(tqdm):
    """Provides `update_to(n)` which uses `tqdm.update(delta_n)`."""

    def update_to(self, b=1, bsize=1, tsize=None):
        """
        b  : int, optional
            Number of blocks transferred so far [default: 1].
        bsize  : int, optional
            Size of each block (in tqdm units) [default: 1].
        tsize  : int, optional
            Total size (in tqdm units). If [default: None] remains unchanged.
        """
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)  # will also set self.n = b * bsize

def get_dataset(dataset_name, target_folder="/", datasets=DATASETS_CONFIG):
    """ Gets the dataset specified by name and return the related components.
    Args:
        dataset_name: string with the name of the dataset
        target_folder (optional): folder to store the datasets, defaults to ./
        datasets (optional): dataset configuration dictionary, defaults to prebuilt one
    Returns:
        img: 3D hyperspectral image (WxHxB)
        gt: 2D int array of labels
        label_values: list of class names
        ignored_labels: list of int classes to ignore
        rgb_bands: int tuple that correspond to red, green and blue bands
    """
    palette = None

    if dataset_name not in datasets.keys():
        raise ValueError("{} dataset is unknown.".format(dataset_name))
    print('when run the def get_dataset, the dataset_name is :',dataset_name)

    # folder = r'E:\2-data\12--xiongan\处理后mat文件' + '/' + dataset_name+ '/'   #     加 个r就是告诉python不要转义
    # folder = target_folder + datasets[dataset_name].get('folder', dataset_name + '/')
    folder = '/home/t1/DL_Code/DATASET/' + dataset_name + '/'
    print('when run the def get_dataset, the data_folder_name is :',folder)

    if dataset_name == 'PaviaC':
        # Load the image
        img = open_file(folder + 'Pavia.mat')
        img = img['pavia']
        rgb_bands = (55, 41, 12)
        gt = open_file(folder + 'Pavia_gt.mat')['pavia_gt']
        label_values = ["Undefined", "Water", "Trees", "Asphalt",
                        "Self-Blocking Bricks", "Bitumen", "Tiles", "Shadows",
                        "Meadows", "Bare Soil"]
        ignored_labels = [0]
    elif dataset_name == 'sd':
        # Load the image
        img = open_file('./Datasets/' + 'add_fog_dataset.mat')
        img = img['Iw']
        rgb_bands = (55, 41, 12)
        gt = open_file('./Datasets/PaviaU/' + 'PaviaU_gt.mat')['paviaU_gt']
        label_values = ['hunkown','Asphalt', 'Meadows', 'Gravel', 'Trees',
                        'Painted metal sheets', 'Bare Soil', 'Bitumen',
                        'Self-Blocking Bricks', 'Shadows']
        ignored_labels = [0]
    elif dataset_name == 'PaviaU':
        # Load the image
        img = open_file(folder + 'PaviaU.mat')
        img = img['paviaU']
        rgb_bands = (55, 41, 12)
        gt = open_file(folder + 'PaviaU_gt.mat')['paviaU_gt']
        label_values = ['Undefined', 'Asphalt', 'Meadows', 'Gravel', 'Trees',
                        'Painted metal sheets', 'Bare Soil', 'Bitumen',
                        'Self-Blocking Bricks', 'Shadows']
        ignored_labels = [0]
    elif dataset_name == 'Salinas':
        img = open_file(folder + 'Salinas_corrected.mat')['salinas_corrected']
        rgb_bands = (43, 21, 11)  # AVIRIS sensor
        gt = open_file(folder + 'Salinas_gt.mat')['salinas_gt']
        label_values = ['Undefined', 'Brocoli_green_weeds_1', 'Brocoli_green_weeds_2', 'Fallow', 'Fallow_rough_plow',
                        'Fallow_smooth', 'Stubble', 'Celery',
                        'Grapes_untrained', 'Soil_vinyard_develop', 'Corn_senesced_green_weeds', 'Lettuce_romaine_4wk',
                        'Lettuce_romaine_5wk', 'Lettuce_romaine_6wk',
                        'Lettuce_romaine_7wk', 'Vinyard_untrained', 'Vinyard_vertical_trellis']
        ignored_labels = [0]
    elif dataset_name == 'MUUFLGulfport':
        # Load the image  1992年Indian Pines数据集
        img = open_file(folder + 'muufl_gulfport_campus_1_hsi.mat')
        img = img['data']
        # img = open_file(folder + 'Indian_pines.mat')
        # img = img['indian_pines']
        rgb_bands = (30, 15, 8)  # AVIRIS sensor
        gt = open_file(folder + 'muufl_gulfport_campus_1_hsi_gt.mat')['gt']
        gt[gt == -1] = 0
        label_values = ["Undefined", "trees", "mostly-grass",
                        "ground surface", "mixed ground surface", "dirt-sand", "road",
                        "water", "buildings", "shadow of buildings",
                        "sidewalk", "yellow curb", "cloth panels"]
        ignored_labels = [0]

    elif dataset_name == 'IndianPines':
        # Load the image  1992年Indian Pines数据集
        img = open_file(folder + 'Indian_pines_corrected.mat')
        img = img['indian_pines_corrected']
        # img = open_file(folder + 'Indian_pines.mat')
        # img = img['indian_pines']
        rgb_bands = (43, 21, 11)  # AVIRIS sensor
        gt = open_file(folder + 'Indian_pines_gt.mat')['indian_pines_gt']
        label_values = ["Undefined", "Alfalfa", "Corn-notill", "Corn-mintill",
                        "Corn", "Grass-pasture", "Grass-trees",
                        "Grass-pasture-mowed", "Hay-windrowed", "Oats",
                        "Soybean-notill", "Soybean-mintill", "Soybean-clean",
                        "Wheat", "Woods", "Buildings-Grass-Trees-Drives",
                        "Stone-Steel-Towers"]
        ignored_labels = [0]
    elif dataset_name == 'Botswana':
        # Load the image
        img = open_file(folder + 'Botswana.mat')['Botswana']
        rgb_bands = (75, 33, 15)
        gt = open_file(folder + 'Botswana_gt.mat')['Botswana_gt']
        label_values = ["Undefined", "Water", "Hippo grass",
                        "Floodplain grasses 1", "Floodplain grasses 2",
                        "Reeds", "Riparian", "Firescar", "Island interior",
                        "Acacia woodlands", "Acacia shrublands",
                        "Acacia grasslands", "Short mopane", "Mixed mopane",
                        "Exposed soils"]
        ignored_labels = [0]
    elif dataset_name == 'ksc':
        # Load the image
        img = open_file(folder + 'ksc.mat')
        img = img['KSC']
        rgb_bands = (43, 21, 11)  # AVIRIS sensor
        gt = open_file(folder + 'ksc_gt.mat')['KSC_gt']
        label_values = ["Undefined", "Scrub", "Willow swamp",
                        "Cabbage palm hammock", "Cabbage palm/oak hammock",
                        "Slash pine", "Oak/broadleaf hammock",
                        "Hardwood swamp", "Graminoid marsh", "Spartina marsh",
                        "Cattail marsh", "Salt marsh", "Mud flats", "Wate"]
        ignored_labels = [0]
    elif dataset_name == 'houston2013':
        # Load the image
        # img = open_file(folder + 'houston.mat')'2013_IEEE_GRSS_DF_Contest_CASI.tif'
        img = open_file('/home/t1/DL_Code/DATASET/houston2013/' + 'Houston.mat')
        img = img['Houston']
        rgb_bands = (59, 40, 23)  # AVIRIS sensor
        gt = open_file('/home/t1/DL_Code/DATASET/houston2013/' + 'Houston_gt.mat')['houston_gt']
        label_values = ["Undefined",
                        "Healthy grass",
                        "Stressed grass",
                        "Synthetic grass",
                        "Trees",
                        "Soil",
                        "Water",
                        "Residential",
                        "Commercial",
                        "Road",
                        "Highway",
                        "Railway",
                        "Parking Lot 1",
                        "Parking Lot 2",
                        "Tennis Court",
                        "Running Track",
                        ]


        # label_values = ["Undefined",
        #                 "Healthy grass",
        #                 "Stressed grass",
        #                 "Synthetic grass",
        #                 "Trees",
        #                 "Soil",
        #                 "Water",
        #                 "Residential",
        #                 "Commercial",
        #                 "Road",
        #                 "Highway",
        #                 "Railway",
        #                 "Parking Lot 1",
        #                 "Parking Lot 2",
        #                 "Tennis Court",
        #                 "Running Track",
        #                 ]
        ignored_labels = [0]
    elif dataset_name == 'houston2018':
        # img = open_file(folder + '2018_IEEE_GRSS_DFC_HSI_TR.HDR')[:, :, :-2]
        img = open_file(folder + 'houston2018.tif')[:, :, :-2]
        print(img.shape)
        # quit()
        # gt = open_file(folder + '2018_IEEE_GRSS_DFC_GT_TR.tif')#[:,0:4172]
        gt = open_file(folder + 'gt.mat')['houston2018_gt']
        print(gt.shape)
        # quit()
        gt = gt.astype('uint8')
        palette = None
        rgb_bands = (47, 31, 15)

        label_values = ["Unclassified",
                        "Healthy grass",
                        "Stressed grass",
                        "Artificial turf",
                        "Evergreen trees",
                        "Deciduous trees",
                        "Bare earth",
                        "Water",
                        "Residential buildings",
                        "Non-residential buildings",
                        "Roads",
                        "Sidewalks",
                        "Crosswalks",
                        "Major thoroughfares",
                        "Highways",
                        "Railways",
                        "Paved parking lots",
                        "Unpaved parking lots",
                        "Cars",
                        "Trains",
                        "Stadium seats"]
        ignored_labels = [0]
    elif dataset_name == 'sd':
        # Load the image
        # img = open_file(folder + 'houston.mat')
        img = open_file(folder + 'Simu_data.mat')
        img = img['Simu_data']
        rgb_bands = (59, 40, 23)  # AVIRIS sensor
        gt = open_file(folder + 'Simu_label.mat')['Simu_label']
        # img = scipy.io.loadmat(folder + 'Simu_data.mat'))['Simu_data']
        # Label = scipy.io.loadmat(os.path.join(DATA_PATH, 'Simu_label.mat'))['Simu_label']

        label_values = ["Undefined",
                        "Healthy grass",
                        "Stressed grass",
                        "Synthetic grass",
                        "Trees",
                        "Soil",
                        ]
        ignored_labels = [0]

        # Data = scipy.io.loadmat(os.path.join(DATA_PATH, 'Simu_data.mat'))['Simu_data']
        # Label = scipy.io.loadmat(os.path.join(DATA_PATH, 'Simu_label.mat'))['Simu_label']

    elif dataset_name == 'xiongan_one':
        img = open_file(folder + 'xiongan_one.mat')
        img = img['xiongan_one']
        gt = open_file(folder + 'Xiongan_one_gt_changed.mat')
        gt = gt['xiongan_one_gt_changed']
        rgb_bands = (120, 72, 36)
        label_values = ["Undefined",
                        "locust tree",
                        "water",
                        "Corn",
                        "pear",
                        "Soybean",
                        "aspen",
                        "vegetable field",
                        "glassland",
                        "peach tree"]
        ignored_labels = [0]
    elif dataset_name == 'xiongan_two':

        img = open_file(folder + 'xiongan_two.mat')
        # print(img)
        img = img['xiongan_two']

        gt = open_file(folder + 'xiongan_two_gt_changed.mat')['xiongan_two_gt_changed']
        rgb_bands = (120, 72, 36)
        label_values = ["Undefined",
                        "paddy",
                        "water",
                        "bare land",
                        "paddy stubble"]
        ignored_labels = [0]
    elif dataset_name == 'xiongan_three':

        img = open_file(folder + 'xiongan_three.mat')
        # print(img)
        img = img['xiongan_three']
        gt = open_file(folder + 'xiongan_three_gt_changed.mat')['xiongan_three_gt_changed']
        rgb_bands = (120, 72, 36)
        label_values = ["Undefined",
                        "pear",
                        "Soybean",
                        "aspen",
                        "glassland"]
        ignored_labels = [0]
    elif dataset_name == 'xiongan_four':

        img = open_file(folder + 'xiongan_four.mat')
        # print(img)
        img = img['xiongan_four']
        gt = open_file(folder + 'xiongan_four_gt_changed.mat')['xiongan_four_gt_changed']
        rgb_bands = (120, 72, 36)
        label_values = ["Undefined",
                        "corn",
                        "pear",
                        "Soybean",
                        "aspen",
                        "vegetable field",
                        "glassland"]
        ignored_labels = [0]


    elif dataset_name == 'WHU-Hi-HanChuan':

        img = open_file(folder + 'WHU_Hi_HanChuan.mat')

        img = img['WHU_Hi_HanChuan']

        gt = open_file(folder + 'WHU_Hi_HanChuan_gt.mat')['WHU_Hi_HanChuan_gt']

        rgb_bands = (110, 70, 33)

        label_values = ["Undefined",

                        " Strawberry",

                        "Cowpea",

                        "Soybean",

                        "Sorghum",

                        "Water spinach",

                        "Watermelon",

                        "Greens",

                        "Trees",

                        "Grass",

                        "Red roof",

                        "Gray roof",

                        "Plastic",

                        "Bare soil",

                        "Road",

                        "Bright object",

                        "Water"]

        ignored_labels = [0]
    elif dataset_name == 'WHU-Hi-HongHu':

        img = open_file(folder + 'WHU_Hi_HongHu.mat')

        img = img['WHU_Hi_HongHu']

        gt = open_file(folder + 'WHU_Hi_HongHu_gt.mat')['WHU_Hi_HongHu_gt']

        rgb_bands = (110, 70, 33)

        label_values = ["Undefined",

                        "Red roof",

                        "Road",

                        "Bare soil",

                        "Cotton",

                        "Cotton firewood",

                        "Rape",

                        "Chinese cabbage",

                        "Pakchoi",

                        "Cabbage",

                        "Tuber mustard",

                        "Brassica parachinensis",

                        "Brassica chinensis",

                        "Small Brassica chinensis",

                        "Lactuca sativa",

                        "Celtuce",

                        "Film covered lettuce",

                        "Romaine lettuce",

                        "Carrot",

                        " White radish",

                        "Garlic sprout",

                        "Broad bean",

                        "Tree"

                        ]

        ignored_labels = [0]
    elif dataset_name == 'WHU-Hi-LongKou':

        img = open_file(folder + 'WHU_Hi_LongKou.mat')

        img = img['WHU_Hi_LongKou']

        gt = open_file(folder + 'WHU_Hi_LongKou_gt.mat')['WHU_Hi_LongKou_gt']

        rgb_bands = (110, 70, 33)

        label_values = ["Undefined",

                        "Corn",

                        "Cotton",

                        "Sesame",

                        "Broad-leaf soybean",

                        "Narrow-leaf soybean",

                        "Rice",

                        "Water",

                        "Roads and houses",

                        "Mixed weed"]

        ignored_labels = [0]

    else:
        # Custom dataset
        img, gt, rgb_bands, ignored_labels, label_values, palette = CUSTOM_DATASETS_CONFIG(dataset_name, folder)
    # Filter NaN out
    nan_mask = np.isnan(img.sum(axis=-1))
    if np.count_nonzero(nan_mask) > 0:
        print("Warning: NaN have been found in the data. It is preferable to remove them beforehand. Learning on NaN data is disabled.")
    img_original = img


    img[nan_mask] = 0
    gt[nan_mask] = 0
    ignored_labels.append(0)
    ignored_labels = list(set(ignored_labels))

    # Normalization
    img = np.asarray(img, dtype='float32')
    img = (img - np.min(img)) / (np.max(img) - np.min(img))
    # print(img[100, 100, :])
    return img,img_original, gt, label_values, ignored_labels, rgb_bands,palette

class HyperX(torch.utils.data.Dataset):
    """ Generic class for a hyperspectral scene """

    def __init__(self, data, gt, **hyperparams):
        """
        Args:
            data: 3D hyperspectral image
            gt: 2D array of labels
            patch_size: int, size of the spatial neighbourhood
            center_pixel: bool, set to True to consider only the label of the
                          center pixel
            data_augmentation: bool, set to True to perform random flips
            supervision: 'full' or 'semi' supervised algorithms
        """
        super(HyperX, self).__init__()
        self.data = data
        self.label = gt
        self.model = hyperparams['model']
        self.name = hyperparams['dataset']
        self.patch_size = hyperparams['patch_size']
        self.ignored_labels = set(hyperparams['ignored_labels'])
        self.flip_augmentation = hyperparams['flip_augmentation']
        self.radiation_augmentation = hyperparams['radiation_augmentation']
        self.mixture_augmentation = hyperparams['mixture_augmentation']
        self.center_pixel = hyperparams['center_pixel']
        supervision = hyperparams['supervision']
        # Fully supervised : use all pixels with label not ignored
        if supervision == 'full':
            mask = np.ones_like(gt)
            mask1_each_class_number = Counter(list(itertools.chain.from_iterable(mask)))
            # print('when run def HyperX,生成一个跟输入 gt 一样大小的全 1 的矩阵，矩阵的元素个数是：', mask1_each_class_number)

            for l in self.ignored_labels:
                mask[gt == l] = 0
            mask2_each_class_number = Counter(list(itertools.chain.from_iterable(mask)))

        elif supervision == 'semi':
            mask = np.ones_like(gt)

        x_pos, y_pos = np.nonzero(mask)  #返回数组 mask 中非零元素的索引值数组

        p = self.patch_size // 2

        self.indices = np.array([(x, y) for x, y in zip(x_pos, y_pos)])
        print(len(self.indices), len(x_pos))

        self.labels = [self.label[x, y] for x, y in self.indices]
        np.random.shuffle(self.indices)

    @staticmethod
    def flip(*arrays):
        horizontal = np.random.random() > 0.5
        vertical = np.random.random() > 0.5
        if horizontal:
            arrays = [np.fliplr(arr) for arr in arrays]
        if vertical:
            arrays = [np.flipud(arr) for arr in arrays]
        return arrays

    def __len__(self):
        # print('when run def __len__(self): the len(self.indices) : ',len(self.indices))
        return len(self.indices)

    def __getitem__(self, i):
        length, width, n_band = self.data.shape[0], self.data.shape[1], self.data.shape[2]

        x, y = self.indices[i]

        x1, y1 = x - self.patch_size // 2, y - self.patch_size // 2
        x2, y2 = x1 + self.patch_size, y1 + self.patch_size

        data = self.data[x1:x2, y1:y2,:]
        label = self.label[x1:x2, y1:y2]

        if self.model in ["dbda"]:
            pass
        else:
            data = np.asarray(np.copy(data).transpose((2, 0, 1)), dtype='float32')

        label = np.asarray(np.copy(label), dtype='int64')
        # Load the data into PyTorch tensors
        data = torch.from_numpy(data)
        # print('when run def HyperX,改变data形状之前的data.size():',data.size())
        label = torch.from_numpy(label)

        if self.center_pixel and self.patch_size > 1:
            label = label[self.patch_size // 2, self.patch_size // 2]

        elif self.patch_size == 1:
            data = data[:, 0, 0]

            label = label[0, 0]

        if self.patch_size > 1:
            #if self.model == '2dcnn':
            if self.model in ["DeformNet_Regular_deform","deepnrd","2dcnn", "2dcnn_plain", "2dcnn_deform",
                              "2dcnn_deform_v2", "2dcnn_deform_epf", "dbda1"]:
               # print('the input model is 2dcnn')
               pass

            else:
                # Make 4D data ((Batch x) Planes x Channels x Width x Height)   (Planes 位面)
                data = data.unsqueeze(0)

        return data, label


