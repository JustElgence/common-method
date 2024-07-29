# -*- coding: utf-8 -*-
import random
import numpy as np
from sklearn.metrics import confusion_matrix
import sklearn.model_selection
import seaborn as sns
import itertools
import spectral
import numpy
import pandas as pd
import matplotlib.pyplot as plt
from scipy import io, misc
import imageio
import os
import re
import torch
from collections import Counter
from sklearn.decomposition import PCA
import cv2 as cv
from skimage.segmentation import slic,mark_boundaries
import time
import logging
def get_device(ordinal):
    # Use GPU
    if ordinal < 0:
        print("Computation on CPU")
        device = torch.device('cpu')
    elif torch.cuda.is_available():
        print("Computation on CUDA GPU device {}".format(ordinal))
        device = torch.device('cuda:{}'.format(ordinal))
    else:
        print("/!\\ CUDA was requested but is not available! Computation will go on CPU. /!\\")
        device = torch.device('cpu')
    return device

def open_file(dataset):
    _, ext = os.path.splitext(dataset)
    ext = ext.lower()
    if ext == '.mat':
        # Load Matlab array
        return io.loadmat(dataset)
    elif ext == '.tif' or ext == '.tiff':
        # Load TIFF file
        return imageio.imread(dataset)
    elif ext == '.hdr':
        img = spectral.open_image(dataset)
        return img.load()

    else:
        raise ValueError("Unknown file format: {}".format(ext))

# 数据可视化用到的函数
def convert_to_color_(arr_2d, palette=None):
    """Convert an array of labels to RGB color-encoded image.

    Args:
        arr_2d: int 2D array of labels
        palette: dict of colors used (label number -> RGB tuple)

    Returns:
        arr_3d: int 2D images of color-encoded labels in RGB format

    """
    arr_3d = np.zeros((arr_2d.shape[0], arr_2d.shape[1], 3), dtype=np.uint8)
    if palette is None:
        raise Exception("Unknown color palette")
    print(palette)
    # quit()
    for c, i in palette.items():
        m = arr_2d == c
        arr_3d[m] = i

    return arr_3d
def convert_from_color_(arr_3d, palette=None):
    """Convert an RGB-encoded image to grayscale labels.

    Args:
        arr_3d: int 2D image of color-coded labels on 3 channels
        palette: dict of colors used (RGB tuple -> label number)

    Returns:
        arr_2d: int 2D array of labels

    """
    if palette is None:
        raise Exception("Unknown color palette")

    arr_2d = np.zeros((arr_3d.shape[0], arr_3d.shape[1]), dtype=np.uint8)

    for c, i in palette.items():
        m = np.all(arr_3d == np.array(c).reshape(1, 1, 3), axis=2)
        arr_2d[m] = i

    return arr_2d
def display_predictions(pred, vis, gt=None, caption=""):
    if gt is None:

        vis.images([np.transpose(pred, (2, 0, 1))],
                    opts={'caption': caption})
    else:

        vis.images([np.transpose(pred, (2, 0, 1)),
                    np.transpose(gt, (2, 0, 1))],
                    nrow=2,
                    opts={'caption': caption})
def display_dataset(img, gt, bands, labels, palette, vis):
    """Display the specified dataset.

    Args:
        img: 3D hyperspectral image
        gt: 2D array labels
        bands: tuple of RGB bands to select
        labels: list of label class names
        palette: dict of colors
        display (optional): type of display, if any

    """
    print("Image has dimensions {}x{} and {} channels".format(*img.shape))
    rgb = spectral.get_rgb(img, bands)
    rgb /= np.max(rgb)
    rgb = np.asarray(255 * rgb, dtype='uint8')

    # Display the RGB composite image
    caption = "RGB (bands {}, {}, {})".format(*bands)
    # send to visdom server
    vis.images([np.transpose(rgb, (2, 0, 1))],
                opts={'caption': caption})
def get_rgb_img(img,bands):

    print("Image has dimensions {}x{} and {} channels".format(*img.shape))
    rgb = spectral.get_rgb(img, bands)
    rgb /= np.max(rgb)
    rgb = np.asarray(255 * rgb, dtype='uint8')
    print("def get_rgb_img(img,bands):数组元素数据类型：", rgb.dtype)  # 打印数组元素数据类型
    print("def get_rgb_img(img,bands):数组形状：", rgb.shape)  # 打印数组形状
    return rgb

def explore_spectrums(img, complete_gt, class_names, vis,ignored_labels=None):
    """Plot sampled spectrums with mean + std for each class.

    Args:
        img: 3D hyperspectral image
        complete_gt: 2D array of labels
        class_names: list of class names
        ignored_labels (optional): list of labels to ignore
        vis : Visdom display
    Returns:
        mean_spectrums: dict of mean spectrum by class

    """
    mean_spectrums = {}
    for c in np.unique(complete_gt):
        if c in ignored_labels:
            continue
        mask = complete_gt == c
        class_spectrums = img[mask].reshape(-1, img.shape[-1])
        step = max(1, class_spectrums.shape[0] // 100)
        fig = plt.figure()
        plt.title(class_names[c])
        # Sample and plot spectrums from the selected class
        for spectrum in class_spectrums[::step, :]:
            plt.plot(spectrum, alpha=0.25)
        mean_spectrum = np.mean(class_spectrums, axis=0)
        std_spectrum = np.std(class_spectrums, axis=0)
        lower_spectrum = np.maximum(0, mean_spectrum - std_spectrum)
        higher_spectrum = mean_spectrum + std_spectrum

        # Plot the mean spectrum with thickness based on std
        plt.fill_between(range(len(mean_spectrum)), lower_spectrum,
                         higher_spectrum, color="#3F5D7D")
        plt.plot(mean_spectrum, alpha=1, color="#FFFFFF", lw=2)
        vis.matplot(plt)
        mean_spectrums[class_names[c]] = mean_spectrum
    return mean_spectrums
def plot_spectrums(spectrums, vis, title=""):
    """Plot the specified dictionary of spectrums.

    Args:
        spectrums: dictionary (name -> spectrum) of spectrums to plot
        vis: Visdom display
    """
    win = None
    for k, v in spectrums.items():
        n_bands = len(v)
        update = None if win is None else 'append'
        win = vis.line(X=np.arange(n_bands), Y=v, name=k, win=win, update=update,
                       opts={'title': title})
# 以上是数据可视化用到的函数

def build_dataset(mat, gt, ignored_labels=None):
    """Create a list of training samples based on an image and a mask.

    Args:
        mat: 3D hyperspectral matrix to extract the spectrums from
        gt: 2D ground truth
        ignored_labels (optional): list of classes to ignore, e.g. 0 to remove
        unlabeled pixels
        return_indices (optional): bool set to True to return the indices of
        the chosen samples

    """
    samples = []
    labels = []
    # Check that image and ground truth have the same 2D dimensions
    assert mat.shape[:2] == gt.shape[:2]

    for label in np.unique(gt):
        if label in ignored_labels:
            continue
        else:
            indices = np.nonzero(gt == label)
            samples += list(mat[indices])
            labels += len(indices[0]) * [label]
    return np.asarray(samples), np.asarray(labels)

def get_random_pos(img, window_shape):
    """ Return the corners of a random window in the input image

    Args:
        img: 2D (or more) image, e.g. RGB or grayscale image
        window_shape: (width, height) tuple of the window

    Returns:
        xmin, xmax, ymin, ymax: tuple of the corners of the window

    """
    w, h = window_shape
    W, H = img.shape[:2]
    x1 = random.randint(0, W - w - 1)
    x2 = x1 + w
    y1 = random.randint(0, H - h - 1)
    y2 = y1 + h
    return x1, x2, y1, y2

def sliding_window(image, test_gt,step, window_size, with_data=True):
    """Sliding window generator over an input image.

    Args:
        image: 2D+ image to slide the window on, e.g. RGB or hyperspectral
        step: int stride of the sliding window
        window_size: int tuple, width and height of the window
        with_data (optional): bool set to True to return both the data and the corner indices  # bool设置为True可同时返回 data 和 the corner indices
    Yields:
        ([data], x, y, w, h) where x and y are the top-left corner of the
        window, (w,h) the window size

    """
    # slide a window across the image
    w, h = window_size
    W, H = image.shape[:2]

    print('when run the def sliding_window, the image.shape[:2] is： ',W,H)
    print('when run the def sliding_window, the window_size is :',w, h)
    # offset_w = (W - w) % step   #   取模 - 返回除法的余数    #  step  default=1,
    # offset_h = (H - h) % step
    # print('when run the def sliding_window, offset_h,offset_w 是： ',offset_h,offset_w)

    for x in range(0, W - w + 1, step):
        if x + w > W:
            x = W - w
        for y in range(0, H - h + 1, step):
            if y + h > H:
                y = H - h
            if test_gt[x,y] != 0:
                if with_data:
                    img = image[x:x + w, y:y + h, :]

                    yield img, x, y, w, h
                else:
                    yield x, y, w, h








def sliding_window_with_test_0(image, test_gt,step, window_size, with_data=True):
    """Sliding window generator over an input image.

    Args:
        image: 2D+ image to slide the window on, e.g. RGB or hyperspectral
        step: int stride of the sliding window
        window_size: int tuple, width and height of the window
        with_data (optional): bool set to True to return both the data and the corner indices  # bool设置为True可同时返回 data 和 the corner indices
    Yields:
        ([data], x, y, w, h) where x and y are the top-left corner of the
        window, (w,h) the window size

    """
    # slide a window across the image
    w, h = window_size
    W, H = image.shape[:2]

    print('when run the def sliding_window, the image.shape[:2] is： ',W,H)
    print('when run the def sliding_window, the window_size is :',w, h)
    # offset_w = (W - w) % step   #   取模 - 返回除法的余数    #  step  default=1,
    # offset_h = (H - h) % step
    # print('when run the def sliding_window, offset_h,offset_w 是： ',offset_h,offset_w)

    for x in range(0, W - w + 1, step):
        if x + w > W:
            x = W - w
        for y in range(0, H - h + 1, step):
            if y + h > H:
                y = H - h
            if with_data:
                # length, width, n_band = image.shape[0], image.shape[1], image.shape[2]
                # if n_band == 107:
                #     img = image[x:x + w, y:y + h, 0:n_band - 1]
                #     # img = image[x:x + w, y:y + h, :]  # 带着 双边滤波结果卷积
                #     # print('test_gt[x,y]:',test_gt[x,y])
                #     if test_gt[x + w // 2, y + h // 2] != 0:
                #
                #         threshold_value = 0.1 # for PaviaU
                #
                #         image_bilateralFilter = image[x:x + w, y:y + h, n_band - 1]
                #         # print('输入 image_bilateralFilter 数据：', image_bilateralFilter)
                #         image_bilateralFilter_central_pixel_abs = abs(
                #             image_bilateralFilter - image_bilateralFilter[w // 2, h // 2])
                #         image_bilateralFilter_central_pixel_abs = np.exp(
                #             -(image_bilateralFilter_central_pixel_abs ** 2) / (0.1))
                #         # threshold_value1 = np.max(
                #         #     [image_bilateralFilter_central_pixel_abs[(w // 2) + 1, h // 2],
                #         #      image_bilateralFilter_central_pixel_abs[(w // 2) - 1, h // 2],
                #         #      image_bilateralFilter_central_pixel_abs[ w // 2, (h // 2) + 1],
                #         #      image_bilateralFilter_central_pixel_abs[ w // 2, (h // 2) - 1]
                #         #      ])
                #         # if threshold_value >= threshold_value1:
                #         #     pass
                #         # else:
                #         #     threshold_value = threshold_value1
                #
                #         # print('输入 image_bilateralFilter_central_pixel_abs 数据：', image_bilateralFilter_central_pixel_abs)
                #         # coordinates_notsame = np.argwhere(image_bilateralFilter > threshold_value)
                #         coordinates_notsame = np.argwhere(image_bilateralFilter_central_pixel_abs < threshold_value )
                #
                #         def calculate_num_the_same_class(biger_size):
                #             # print('biger_size', biger_size)
                #
                #             # print('threshold_value: ',threshold_value)
                #
                #             patch_size1 = w
                #             central_piexl_coo_x, central_piexl_coo_y = x + w // 2, y + h // 2
                #
                #             x1, y1 = x, y
                #             x2, y2 = x1 + patch_size1, y1 + patch_size1
                #
                #             x_begin = x1 - biger_size
                #             x_end = x2 + biger_size
                #             y_begin = y1 - biger_size
                #             y_end = y2 + biger_size
                #             if x_begin >= 0 and x_end <= length and y_begin >= 0 and y_end <= width:
                #                 pass
                #             else:
                #                 if x_begin < 0:
                #                     x_begin = 0
                #                 if x_end > length:
                #                     x_end = length
                #                 if y_begin < 0:
                #                     y_begin = 0
                #                 if y_end > width:
                #                     y_end = width
                #
                #             # print('x_begin:x_end, y_begin:y_end',x_begin,x_end, y_begin,y_end,x1, y1)
                #             image_bilateralFilter_out = image[x_begin:x_end, y_begin:y_end, n_band - 1]
                #             # print('输入 image_bilateralFilter_out 数据：', image_bilateralFilter_out)
                #             image_bilateralFilter_out_central_pixel_abs = abs(
                #                 image_bilateralFilter_out - image[central_piexl_coo_x, central_piexl_coo_y, n_band - 1])
                #             # print('输入 image_bilateralFilter_out_central_pixel_abs 数据：', image_bilateralFilter_out_central_pixel_abs)
                #             image_bilateralFilter_out_central_pixel_abs = np.exp(
                #                 -(image_bilateralFilter_out_central_pixel_abs ** 2) / (0.1))
                #
                #             coordinates_out_notsame = np.argwhere(
                #                 image_bilateralFilter_out_central_pixel_abs < threshold_value)
                #             coordinates_out_same = np.argwhere(
                #                 image_bilateralFilter_out_central_pixel_abs > threshold_value)
                #
                #             # print('\n', 'coordinates_out_notsame: ', '\n', coordinates_out_notsame, '\n', 'coordinates_out_same: ', '\n',coordinates_out_same)
                #
                #             if biger_size <= 5:
                #                 # print('biger_size:', biger_size)
                #                 if len(coordinates_out_same) >= patch_size1 * patch_size1:
                #                     image_bilateralFilter_out_central_pixel_abs1 = image_bilateralFilter_out_central_pixel_abs - np.pad(
                #                         image_bilateralFilter_central_pixel_abs,
                #                         ((x1 - x_begin, x_end - x2), (y1 - y_begin, y_end - y2)),
                #                         'constant', constant_values=(0, 0))
                #                     # print('输入 image_bilateralFilter_out_central_pixel_abs1 数据：',
                #                     #       image_bilateralFilter_out_central_pixel_abs1)
                #                     image_bilateralFilter_out_central_pixel_abs_flatten = image_bilateralFilter_out_central_pixel_abs1.flatten()
                #                     image_bilateralFilter_out_central_pixel_abs_flatten.sort()
                #                     # print('输入 image_bilateralFilter_out_central_pixel_abs_flatten 数据：',
                #                     #       image_bilateralFilter_out_central_pixel_abs_flatten)
                #
                #                     return coordinates_out_same, x_begin, y_begin, \
                #                            image_bilateralFilter_out_central_pixel_abs_flatten, image_bilateralFilter_out_central_pixel_abs
                #                 else:
                #                     return calculate_num_the_same_class(biger_size + 1)
                #             else:
                #                 # print('biger_size:',biger_size)
                #                 image_bilateralFilter_out_central_pixel_abs1 = image_bilateralFilter_out_central_pixel_abs - np.pad(
                #                     image_bilateralFilter_central_pixel_abs,
                #                     ((x1 - x_begin, x_end - x2), (y1 - y_begin, y_end - y2)),
                #                     'constant', constant_values=(0, 0))
                #                 # print('输入 image_bilateralFilter_out_central_pixel_abs1 数据：',
                #                 #       image_bilateralFilter_out_central_pixel_abs1)
                #                 image_bilateralFilter_out_central_pixel_abs_flatten = image_bilateralFilter_out_central_pixel_abs1.flatten()
                #                 image_bilateralFilter_out_central_pixel_abs_flatten.sort()
                #                 # print('输入 image_bilateralFilter_out_central_pixel_abs_flatten 数据：',
                #                 #       image_bilateralFilter_out_central_pixel_abs_flatten)
                #                 return coordinates_out_same, x_begin, y_begin, \
                #                        image_bilateralFilter_out_central_pixel_abs_flatten, image_bilateralFilter_out_central_pixel_abs
                #
                #         if len(coordinates_notsame) == 0:
                #             # print('不需要可变卷积')
                #             pass
                #         # patch_size*patch_size 内部需要替换的个数 小于等于 外部可以选择的个数
                #         else:
                #             # print('需要可变卷积')
                #             biger_size = 1
                #
                #             coordinates_out_same, x_begin, y_begin, \
                #             image_bilateralFilter_out_central_pixel_abs_flatten, image_bilateralFilter_out_central_pixel_abs = \
                #                 calculate_num_the_same_class(biger_size)
                #             # coordinates_out_same, x_begin, y_begin, image_bilateralFilter_out_central_pixel_abs = \
                #             #     calculate_num_the_same_class_gt(biger_size)
                #             # print(image_bilateralFilter_out_central_pixel_abs.shape)
                #             # quit()
                #
                #             # print('\n', 'coordinates_out_same: ', '\n', coordinates_out_same)
                #             # print('输入 image_bilateralFilter_out_central_pixel_abs_flatten 数据：',
                #             #       image_bilateralFilter_out_central_pixel_abs_flatten)
                #             # print('输入 image_bilateralFilter_out_central_pixel_abs 数据：', image_bilateralFilter_out_central_pixel_abs)
                #
                #             # coordinates1 = np.argwhere(
                #             #     image_bilateralFilter_out_central_pixel_abs == image_bilateralFilter[w // 2, h // 2])
                #
                #             # print(coordinates1)
                #             # print('kkkk')
                #             # print(coordinates_notsame)
                #             # print(x_begin, y_begin)
                #             # quit()
                #
                #             num0 = len(image_bilateralFilter_out_central_pixel_abs_flatten)
                #             num_need_to_replace = len(coordinates_notsame)
                #             # print('len(coordinates_out_same)',len(coordinates_out_same),'num_need_to_replace: ',num_need_to_replace,
                #             #       'num0: ',num0,'biger_size1: ',biger_size1)
                #
                #             for m in range(num_need_to_replace):
                #                 coordinates1 = np.argwhere(
                #                     image_bilateralFilter_out_central_pixel_abs ==
                #                     image_bilateralFilter_out_central_pixel_abs_flatten[num0-m-1])
                #
                #                 # actual_coo_out_the_same_class_x = coordinates1[0][0] - image_bilateralFilter_out_central_pixel_abs.shape[0] // 2 + central_piexl_coo_x
                #                 # actual_coo_out_the_same_class_y = coordinates1[0][1] - image_bilateralFilter_out_central_pixel_abs.shape[1] // 2 + central_piexl_coo_y
                #                 actual_coo_out_the_same_class_x = x_begin + coordinates1[0][0]
                #                 actual_coo_out_the_same_class_y = y_begin + coordinates1[0][1]
                #                 # actual_coo_out_the_same_class_x = x_begin + coordinates1[m][0]
                #                 # actual_coo_out_the_same_class_y = y_begin + coordinates1[m][1]
                #
                #                 coordinates2 = coordinates_notsame[m][0]
                #                 coordinates3 = coordinates_notsame[m][1]
                #                 #
                #                 # print(coordinates1, '\n', 'coordinates2,coordinates3: ', '\n', coordinates2, coordinates3, '\n',
                #                 #       actual_coo_out_the_same_class_x,actual_coo_out_the_same_class_y)
                #                 # print('data.shape', data.shape)
                #                 # img[coordinates2, coordinates3, :] = 0
                #                 img[coordinates2, coordinates3, :] = image[actual_coo_out_the_same_class_x,
                #                                                      actual_coo_out_the_same_class_y, 0:n_band - 1]
                #                 # img[coordinates2, coordinates3, :] = image[actual_coo_out_the_same_class_x,
                #                 #                                      actual_coo_out_the_same_class_y,:]  # 带着 双边滤波结果卷积
                #                 # print('data.shape', data.shape)
                # else:
                img = image[x:x + w, y:y + h, :]

                yield img, x, y, w, h
            else:
                yield x, y, w, h

def count_sliding_window(top, test_gt,step, window_size):
    """ Count the number of windows in an image.

    Args:
        image: 2D+ image to slide the window on, e.g. RGB or hyperspectral, ...
        step: int stride of the sliding window
        window_size: int tuple, width and height of the window
    Returns:
        int number of windows
    """
    sw = sliding_window(top, test_gt,step, window_size, with_data=False)

    #quit()
    return sum(1 for _ in sw)

def grouper(n, iterable):
    """ Browse an iterable by grouping n elements by n elements.

    Args:
        n: int, size of the groups
        iterable: the iterable to Browse
    Yields:
        chunk of n elements from the iterable

    """
    it = iter(iterable)
    while True:
        chunk = tuple(itertools.islice(it, n))
        if not chunk:
            return
        yield chunk

def metrics1(prediction, target,hyperparams,n_classes=None):
    """Compute and print metrics (accuracy, confusion matrix and F1 scores).

    Args:
        prediction: list of predicted labels
        target: list of target labels
        ignored_labels (optional): list of labels to ignore, e.g. 0 for undef
        n_classes (optional): number of classes, max(target) by default
    Returns:
        accuracy, F1 score by class, confusion matrix
    """
    print('***************************分隔符**************************')
    print(' run def metrics')
    print('***************************分隔符**************************')
    print('when run def metrics , 输入高光谱数据的维度：', target.shape[0], target.shape[1])

    center_pixel = hyperparams['center_pixel']
    patch_size= hyperparams['patch_size']
    ignored_labels = hyperparams['ignored_labels']
    # 产生截取坐标
    img_abscissa = target.shape[0]
    img_ordinate = target.shape[1]
    abscissa_begin = patch_size // 2
    ordinate_begin = patch_size // 2
    abscissa_end = img_abscissa - patch_size  + abscissa_begin
    ordinate_end = img_ordinate - patch_size  + ordinate_begin
    # print('when run def metrics ,产生的截取坐标是abscissa_begin,abscissa_end,ordinate_begin,ordinate_end is :',abscissa_begin,abscissa_end,ordinate_begin,ordinate_end)

    ignored_mask = np.zeros(target.shape[:2], dtype=np.bool)
    # print("when run def metrics ,生成的全 0 的 ignored_mask 数据类型", type(ignored_mask))      # 打印数组数据类型
    # print("when run def metrics ,生成的全 0 的 ignored_mask 数组元素总数：", ignored_mask.size)  # 打印数组尺寸，即数组元素总数
    # print("when run def metrics ,生成的全 0 的 ignored_mask 数组形状：", ignored_mask.shape)    # 打印数组形状
    # print("when run def metrics ,生成的全 0 的 ignored_mask 数组内容：", '\n',ignored_mask)
    # print("when run def metrics ,ignored_labels 的内容：",'\n', ignored_labels)

    for l in ignored_labels:
        # print('when run def metrics, l is : ',l)

        ignored_mask[target == l] = True   #  将 target 值是 0 的位置  ignored_mask 在这个位置是 true
        # print("when run def metrics ,将 target 值是 0 的位置的 ignored_mask 的值改成 true 的 ignored_mask 数据类型", type(ignored_mask))      # 打印数组数据类型
        # print("when run def metrics ,将 target 值是 0 的位置的 ignored_mask 的值改成 true 的 ignored_mask 数组元素总数：", ignored_mask.size)  # 打印数组尺寸，即数组元素总数
        # print("when run def metrics ,将 target 值是 0 的位置的 ignored_mask 的值改成 true 的 ignored_mask 数组形状：", ignored_mask.shape)    # 打印数组形状
        # print('when run def metrics ,将 target 值是 0 的位置的 ignored_mask 的值改成 true 的 ignored_mask 内容：','\n',ignored_mask)

    ignored_mask = ~ignored_mask

    if center_pixel:
        ignored_mask = ignored_mask[abscissa_begin:abscissa_end,ordinate_begin:ordinate_end]
    else:
        ignored_mask = ignored_mask

    # print('when run def metrics ,取反后的 ignored_mask ：','\n',ignored_mask)   # target （1-9）标签的位置置 true ，（0）的位置置 false
    # print('when run def metrics ,未处理的 target 即 test_gt ：','\n',target)
    # target_each_class_number = Counter(list(itertools.chain.from_iterable(target)))
    # print('when run def metrics ,未处理的 target 即 test_gt 的 各类别数量： ',target_each_class_number, len(list(itertools.chain.from_iterable(target))))
    # print('***************************分隔符**************************')

    # 该写法的目的应该是 将 [ ] 里 是 true 的对应位置的 target 的值保留， [ ] 里 是 false 的对应位置的 target 的值丢弃
    if center_pixel:
        target = target[abscissa_begin:abscissa_end,ordinate_begin:ordinate_end]
    else:
        target = target

    target = target[ignored_mask]

    np.set_printoptions(threshold=np.inf)
    # print('when run def metrics ,处理后的 target 的shape：',target.shape)       # target_gt 中不是 0 的 个数 。即 带标签的数量
    # print('when run def metrics ,处理后的 target 的 各元素个数 ：',Counter(target))
    # print('***************************分隔符**************************')
    # print("when run def metrics ,未处理的 prediction 数组元素总数：", prediction.size)  # 打印数组尺寸，即数组元素总数
    # print("when run def metrics ,未处理的 prediction 数组形状：", prediction.shape)    # 打印数组形状


    if center_pixel:
        prediction = prediction[abscissa_begin:abscissa_end,ordinate_begin:ordinate_end]
    else:
        prediction = prediction

    # 该写法的目的应该是 将 [ ] 里 是 true 的对应位置的 target 的值保留， [ ] 里 是 false 的对应位置的 target 的值丢弃
    prediction = prediction[ignored_mask]
    # print('when run def metrics ,处理后的 prediction 的 shape ：', prediction.shape)
    # print('when run def metrics ,处理后的 prediction 的各元素个数 ：',Counter(prediction),prediction.size)

    results = {}
    n_classes = np.max(target) + 1 if n_classes is None else n_classes
    # print('when run def metrics , n_classes :',n_classes)
    numpy.set_printoptions(linewidth=300)

    cm = confusion_matrix(
        target,
        prediction,
        labels=range(n_classes))

    results["Confusion matrix"] = cm

    # Compute global accuracy   OA
    total = np.sum(cm)
    accuracy = sum([cm[x][x] for x in range(len(cm))])
    accuracy *= 100 / float(total)

    results["Accuracy"] = accuracy


    # Compute  class accuracy    类别准确率CA 所有类别准确率的均值AA

    class_accuracy = [cm[x][x] / sum(cm[x]) for x in range(len(cm))]

    results["Class_Accuracy"] = class_accuracy

    # Compute F1 score
    F1scores = np.zeros(len(cm))
    for i in range(len(cm)):
        try:
            F1 = 2. * cm[i, i] / (np.sum(cm[i, :]) + np.sum(cm[:, i]))
        except ZeroDivisionError:
            F1 = 0.
        F1scores[i] = F1

    results["F1 scores"] = F1scores

    # Compute kappa coefficient
    pa = np.trace(cm) / float(total)
    pe = np.sum(np.sum(cm, axis=0) * np.sum(cm, axis=1)) / \
        float(total * total)
    kappa = (pa - pe) / (1 - pe)
    results["Kappa"] = kappa

    return results

def setup_logger(logger_name, root, level=logging.INFO, screen=False, tofile=False):
    """set up logger"""
    lg = logging.getLogger(logger_name)
    formatter = logging.Formatter("[%(asctime)s.%(msecs)03d] %(message)s", datefmt="%H:%M:%S")
    lg.setLevel(level)
    if tofile:
        log_file = os.path.join(root, "{}_{}.log".format(logger_name, get_timestamp()))
        fh = logging.FileHandler(log_file, mode="w")
        fh.setFormatter(formatter)
        lg.addHandler(fh)
    if screen:
        sh = logging.StreamHandler()
        sh.setFormatter(formatter)
        lg.addHandler(sh)
    return lg

def get_timestamp():
    timestampTime = time.strftime("%H%M%S")
    timestampDate = time.strftime("%Y%m%d")
    return timestampDate + "-" + timestampTime

def metrics(prediction, target, hyperparams, n_classes=None):
    """Compute and print metrics (accuracy, confusion matrix and F1 scores).
    Args:
        prediction: list of predicted labels
        target: list of target labels
        ignored_labels (optional): list of labels to ignore, e.g. 0 for undef
        n_classes (optional): number of classes, max(target) by default
    Returns:
        accuracy, F1 score by class, confusion matrix
    """
    ignored_mask = np.zeros(target.shape[:2], dtype=np.bool)
    ignored_labels = hyperparams['ignored_labels']
    for l in ignored_labels:
        ignored_mask[target == l] = True
    ignored_mask = ~ignored_mask
    target = target[ignored_mask]
    prediction = prediction[ignored_mask]

    results = {}

    n_classes = np.max(target) + 1 if n_classes is None else n_classes
    # print(n_classes)

    cm = confusion_matrix(
        target,
        prediction,
        labels=range(n_classes))
    # print(cm)
    cm = cm[1:,1:]
    # print(cm)

    results["Confusion matrix"] = cm

    # Compute global accuracy
    total = np.sum(cm)
    accuracy = sum([cm[x][x] for x in range(len(cm))])
    accuracy *= 100 / float(total)

    results["Accuracy"] = accuracy

    # Compute  class accuracy    类别准确率CA 所有类别准确率的均值AA
    class_accuracy = [cm[x][x] / sum(cm[x]) for x in range(len(cm))]
    results["Class_Accuracy"] = np.mean(class_accuracy)

    # Compute F1 score
    F1scores = np.zeros(len(cm))
    for i in range(len(cm)):
        try:
            F1 = 2. * cm[i, i] / (np.sum(cm[i, :]) + np.sum(cm[:, i]))
        except ZeroDivisionError:
            F1 = 0.
        F1scores[i] = F1

    results["F1 scores"] = F1scores

    # Compute kappa coefficient
    pa = np.trace(cm) / float(total)
    pe = np.sum(np.sum(cm, axis=0) * np.sum(cm, axis=1)) / \
        float(total * total)
    kappa = (pa - pe) / (1 - pe)
    results["Kappa"] = kappa
    # print(len(cm))

    for i in range(len(cm)):
        print(class_accuracy[i])
    # print([class_accuracy[i] for i in range(len(cm))])
    # quit()

    #
    # print(class_accuracy[0])
    # print(class_accuracy[1])
    # print(class_accuracy[2])
    # print(class_accuracy[3])
    # print(class_accuracy[4])
    # print(class_accuracy[5])
    # print(class_accuracy[6])
    # print(class_accuracy[7])
    # print(class_accuracy[8])
    # print(class_accuracy[9])
    # print(class_accuracy[10])
    # print(class_accuracy[11])
    # print(class_accuracy[12])
    # print(class_accuracy[13])
    # print(class_accuracy[14])
    print(accuracy / 100)
    print(np.mean(class_accuracy))
    print(kappa)

    return results
def show_results(SAMPLE_PERCENTAGE,hyperparams,results, vis, label_values=None, agregated=False):

    patch_size = hyperparams['patch_size']
    model = hyperparams['model']
    dataset_name = hyperparams['dataset']
    batch_size = hyperparams['batch_size']

    text = ""
    if agregated:
        accuracies = [r["Accuracy"] for r in results]
        kappas = [r["Kappa"] for r in results]
        F1_scores = [r["F1 scores"] for r in results]

        F1_scores_mean = np.mean(F1_scores, axis=0)
        F1_scores_std = np.std(F1_scores, axis=0)
        cm = np.mean([r["Confusion matrix"] for r in results], axis=0)
        text += "Agregated results :\n"
    else:
        cm = results["Confusion matrix"]
        accuracy = results["Accuracy"]
        F1scores = results["F1 scores"]
        kappa = results["Kappa"]
        class_accuracy = results["Class_Accuracy"]


    label_values = label_values[1:]

    vis.heatmap(cm, opts={'title': "Confusion matrix",
                          'marginbottom': 150,
                          'marginleft': 150,
                          'width': 500,
                          'height': 500,
                          'rownames': label_values, 'columnnames': label_values})
    text += "patch_size :"
    text += str(patch_size)
    text += "\n"
    text += "model :"
    text += str(model)
    text += "\n"
    text += "dataset_name :"
    text += str(dataset_name)
    text += "\n"
    text += "batch_size :"
    text += str(batch_size)
    text += "\n"
    text += "SAMPLE_PERCENTAGE :"
    text += str(SAMPLE_PERCENTAGE)
    text += "\n"


    text += "Confusion matrix :\n"
    text += str(cm)
    text += "---\n"

    if agregated:
        text += ("Accuracy: {:.02f} +- {:.02f}\n".format(np.mean(accuracies)*100,
                                                         np.std(accuracies)*100))
    else:
        text += "Accuracy : {:.04f}%\n".format(accuracy)

        text += "class_accuracy:"
        text += str(class_accuracy)

    text += "---\n"

    text += "F1 scores :\n"
    if agregated:
        for label, score, std in zip(label_values, F1_scores_mean,
                                     F1_scores_std):
            text += "\t{}: {:.02f} +- {:.02f}\n".format(label, score*100, std*100)
    else:
        for label, score in zip(label_values, F1scores):
            text += "\t{}: {:.03f}\n".format(label, score)
    text += "---\n"

    if agregated:
        text += ("Kappa: {:.02f} +- {:.02f}\n".format(np.mean(kappas)*100,
                                                      np.std(kappas)*100))
    else:
        text += "Kappa: {:.03f}\n".format(kappa)

    vis.text(text.replace('\n', '<br/>'))
    print('text is :',text)

    b = os.getcwd() + '/result/'
    if not os.path.exists(b):  # 判断当前路径是否存在，没有则创建new文件夹
        os.makedirs(b)
    # mode 模式
    # w 只能操作写入  r 只能读取 a 向文件追加
    # w+ 可读可写 r+可读可写  a+可读可追加
    # wb+写入进制数据
    # w模式打开文件，如果而文件中有数据，再次写入内容，会把原来的覆盖掉

    file_name = b + "classification_report_" + dataset_name + "_" + model + "_" + str(patch_size) + ".txt"
    file_handle = open(file_name, mode='a')

    file_handle.write('\n')
    file_handle.write(text)
    file_handle.write('\n')
    file_handle.close()
    print('混淆矩阵中,一行求和总数是该类别的真实数量，一列求和就是该类别的预测数量')
def show_results1(results, vis, label_values=None, agregated=False):

    text = ""
    if agregated:
        accuracies = [r["Accuracy"] for r in results]
        kappas = [r["Kappa"] for r in results]
        F1_scores = [r["F1 scores"] for r in results]

        F1_scores_mean = np.mean(F1_scores, axis=0)
        F1_scores_std = np.std(F1_scores, axis=0)
        cm = np.mean([r["Confusion matrix"] for r in results], axis=0)
        text += "Agregated results :\n"
    else:
        cm = results["Confusion matrix"]
        accuracy = results["Accuracy"]
        F1scores = results["F1 scores"]
        kappa = results["Kappa"]
        class_accuracy = results["Class_Accuracy"]


    label_values = label_values[1:]

    vis.heatmap(cm, opts={'title': "Confusion matrix",
                          'marginbottom': 150,
                          'marginleft': 150,
                          'width': 500,
                          'height': 500,
                          'rownames': label_values, 'columnnames': label_values})

    text += "Confusion matrix :\n"
    text += str(cm)
    text += "---\n"

    if agregated:
        text += ("Accuracy: {:.03f} +- {:.03f}\n".format(np.mean(accuracies),
                                                         np.std(accuracies)))
    else:
        text += "Accuracy : {:.03f}%\n".format(accuracy)

        text += "class_accuracy:"
        text += str(class_accuracy)

    text += "---\n"

    text += "F1 scores :\n"
    if agregated:
        for label, score, std in zip(label_values, F1_scores_mean,
                                     F1_scores_std):
            text += "\t{}: {:.03f} +- {:.03f}\n".format(label, score, std)
    else:
        for label, score in zip(label_values, F1scores):
            text += "\t{}: {:.03f}\n".format(label, score)
    text += "---\n"

    if agregated:
        text += ("Kappa: {:.03f} +- {:.03f}\n".format(np.mean(kappas),
                                                      np.std(kappas)))
    else:
        text += "Kappa: {:.03f}\n".format(kappa)

    vis.text(text.replace('\n', '<br/>'))
    print('text is :',text)
    print('混淆矩阵中,一行求和总数是该类别的真实数量，一列求和就是该类别的预测数量')
def sample_gt(gt, train_size, mode):
    """Extract a fixed percentage of samples from an array of labels.

    Args:
        gt: a 2D array of int labels
        percentage: [0, 1] float
    Returns:
        train_gt, test_gt: 2D arrays of int labels

    """
    indices = np.nonzero(gt)
    X = list(zip(*indices)) # x,y features
    y = gt[indices].ravel() # classes
    train_gt = np.zeros_like(gt)
    test_gt = np.zeros_like(gt)

    if train_size > 1:
       train_size = int(train_size)  # int() 函数用于将一个字符串或数字转换为整型。x -- 字符串或数字。 base -- 进制数，默认十进制。
    
    if mode == 'random':
       print("当选取样本的时候，Sampling {} with train size = {}".format(mode, train_size))

       train_indices, test_indices = sklearn.model_selection.train_test_split(X, train_size=train_size, stratify=y)
       train_indices = [list(t) for t in zip(*train_indices)]
       test_indices = [list(t) for t in zip(*test_indices)]
       train_gt[tuple(train_indices)] = gt[tuple(train_indices)]
       test_gt[tuple(test_indices)] = gt[tuple(test_indices)]

    elif mode == 'fixed':
        # 选择 固定数量样本
       '''
        sample_gt 函数输出的 train_gt 中各样本数量：  Counter({0: 206500, 1: 100, 4: 100, 2: 100, 8: 100, 5: 100, 9: 100, 6: 100, 3: 100, 7: 100}) 207400
        sample_gt 函数输出的 test_gt 中各样本数量：  Counter({0: 165524, 2: 18549, 1: 6531, 6: 4929, 8: 3582, 4: 2964, 3: 1999, 5: 1245, 7: 1230, 9: 847}) 207400   
       '''
       print("当选取样本的时候，Sampling {} with train size = {}".format(mode, train_size))
       train_indices, test_indices = [], []
           #    train_size = [100,10,10,10,10,10, 10,10,10,10]
       train_size = [100,10,10,10,10,10, 10,10,10,10,10, 10,10,10,10,10,10]
       #train_size = [100,500,500,68,500,500,451,26,500,500,500,500,151,500,500,500,500,14,500,500,500]
    #    train_size = [100,5,5,5,5,5, 5,5,5,5]
    #    train_size = [100,5,5,5,5,5, 5,5,5,5,5,5,5 ,5 ,5 ,5,5]
    #    train_size = [100,1,1,1,1,1, 1,1,1,1]
    #    train_size = [100,2,2,2,2,2, 2,2,2,2]
    #    train_size = [100,3,3,3,3,3, 3,3,3,3]
    #    train_size = [100,4,4,4,4,4, 4,4,4,4]
       for c in np.unique(gt):
           if c == 0:
              continue  #   continue 语句用来告诉Python跳过当前循环的剩余语句，然后继续进行下一轮循环。
           indices = np.nonzero(gt == c)
           X = list(zip(*indices)) # x,y features

           train, test = sklearn.model_selection.train_test_split(X, train_size=train_size[c])
           train_indices += train
           test_indices += test
       train_indices = [list(t) for t in zip(*train_indices)]
       test_indices = [list(t) for t in zip(*test_indices)]
       train_gt[tuple(train_indices)] = gt[tuple(train_indices)]
       test_gt[tuple(test_indices)] = gt[tuple(test_indices)]

    elif mode == 'disjoint':
        print("当选取样本的时候，Sampling {} with train size = {}".format(mode, train_size))
        train_gt = np.copy(gt)
        test_gt = np.copy(gt)
        for c in np.unique(gt):
            mask = gt == c
            for x in range(gt.shape[0]):
                first_half_count = np.count_nonzero(mask[:x, :])
                second_half_count = np.count_nonzero(mask[x:, :])
                try:
                    ratio = first_half_count / second_half_count
                    if ratio > 0.9 * train_size and ratio < 1.1 * train_size:
                        break
                except ZeroDivisionError:
                    continue
            mask[:x, :] = 0
            train_gt[mask] = 0

        test_gt[train_gt > 0] = 0
    else:
        raise ValueError("{} sampling is not implemented yet.".format(mode))
    return train_gt, test_gt

def compute_imf_weights(ground_truth, n_classes=None, ignored_classes=[]):
    """ Compute inverse median frequency weights for class balancing.

    For each class i, it computes its frequency f_i, i.e the ratio between
    the number of pixels from class i and the total number of pixels.

    Then, it computes the median m of all frequencies. For each class the
    associated weight is m/f_i.

    Args:
        ground_truth: the annotations array
        n_classes: number of classes (optional, defaults to max(ground_truth))
        ignored_classes: id of classes to ignore (optional)
    Returns:
        numpy array with the IMF coefficients 
    """
    n_classes = np.max(ground_truth) if n_classes is None else n_classes
    weights = np.zeros(n_classes)
    frequencies = np.zeros(n_classes)

    for c in range(0, n_classes):
        if c in ignored_classes:
            continue
        frequencies[c] = np.count_nonzero(ground_truth == c)

    # Normalize the pixel counts to obtain frequencies
    frequencies /= np.sum(frequencies)
    # Obtain the median on non-zero frequencies
    idx = np.nonzero(frequencies)
    median = np.median(frequencies[idx])
    weights[idx] = median / frequencies[idx]
    weights[frequencies == 0] = 0.
    return weights

def camel_to_snake(name):
    s = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s).lower()

def pca_img(x,x_width,x_length,x_bands,n_components):
    Data = x
    m, n, l = x_width, x_length, x_bands
    # extract the first principal component
    x = np.reshape(Data, (m * n, l))
    pca = PCA(n_components, copy=True, whiten=False)
    x = pca.fit_transform(x)
    _, l = x.shape
    x = np.reshape(x, (m, n, l))
    #print(x.shape)
    return x


def bilateralFilter_rgb_img(src, diameter, sigmaColor, sigmaSpace):
    rgb_img_bilateralFilter = cv.bilateralFilter(src, diameter, sigmaColor,
                                                 sigmaSpace)  # src, diameter, sigmaColor, sigmaSpace
    print("def bilateralFilter 输出 rgb_img_bilateralFilter 数组元素数据类型：", rgb_img_bilateralFilter.dtype)  # 打印数组元素数据类型
    print("def bilateralFilter 输出 rgb_img_bilateralFilter 数组元素总数：", rgb_img_bilateralFilter.size)  # 打印数组尺寸，即数组元素总数
    print("def bilateralFilter 输出 rgb_img_bilateralFilter 数组形状：", rgb_img_bilateralFilter.shape)  # 打印数组形状
    print("def bilateralFilter 输出 rgb_img_bilateralFilter 数组的维度数目", rgb_img_bilateralFilter.ndim)  # 打印数组的维度数目

    rgb_img_bilateralFilter_sum = np.sum(rgb_img_bilateralFilter, axis=2)
    print("def bilateralFilter 输出 rgb_img_bilateralFilter_sum 数组元素数据类型：",
          rgb_img_bilateralFilter_sum.dtype)  # 打印数组元素数据类型
    print("def bilateralFilter 输出 rgb_img_bilateralFilter_sum 数组元素总数：",
          rgb_img_bilateralFilter_sum.size)  # 打印数组尺寸，即数组元素总数
    print("def bilateralFilter 输出 rgb_img_bilateralFilter_sum 数组形状：", rgb_img_bilateralFilter_sum.shape)  # 打印数组形状
    print("def bilateralFilter 输出 rgb_img_bilateralFilter_sum 数组的维度数目", rgb_img_bilateralFilter_sum.ndim)  # 打印数组的维度数目

    return rgb_img_bilateralFilter_sum

def bilateralFilter_img(src, diameter, sigmaColor, sigmaSpace):
    # 将输入进来的 img 进行降维

    img_pca = pca_img(src, src.shape[0], src.shape[1], src.shape[2], 3)

    img_bilateralFilter = cv.bilateralFilter(img_pca, diameter, sigmaColor,
                                             sigmaSpace)  # src, diameter, sigmaColor, sigmaSpace
    # print("def bilateralFilter 输出 rgb_img_bilateralFilter 数组元素数据类型：", img_bilateralFilter.dtype)  # 打印数组元素数据类型
    # print("def bilateralFilter 输出 rgb_img_bilateralFilter 数组元素总数：", img_bilateralFilter.size)  # 打印数组尺寸，即数组元素总数
    # print("def bilateralFilter 输出 rgb_img_bilateralFilter 数组形状：", img_bilateralFilter.shape)  # 打印数组形状
    # print("def bilateralFilter 输出 rgb_img_bilateralFilter 数组的维度数目", img_bilateralFilter.ndim)  # 打印数组的维度数目

    img_bilateralFilter_sum = np.sum(img_bilateralFilter, axis=2)
    # print("def bilateralFilter 输出 rgb_img_bilateralFilter_sum 数组元素数据类型：",
    #       img_bilateralFilter_sum.dtype)  # 打印数组元素数据类型
    # print("def bilateralFilter 输出 rgb_img_bilateralFilter_sum 数组元素总数：",
    #       img_bilateralFilter_sum.size)  # 打印数组尺寸，即数组元素总数
    # print("def bilateralFilter 输出 rgb_img_bilateralFilter_sum 数组形状：", img_bilateralFilter_sum.shape)  # 打印数组形状
    # print("def bilateralFilter 输出 rgb_img_bilateralFilter_sum 数组的维度数目", img_bilateralFilter_sum.ndim)  # 打印数组的维度数目


    return img_bilateralFilter_sum

def bilateral_filter1(image, gsigma, ssigma, winsize):

    r = int(winsize / 2)
    c = r
    '''
    #在数组A的边缘填充constant_values指定的数值
    #（3,2）表示在A的第[0]轴填充（二维数组中，0轴表示行），即在0轴前面填充3个宽度的0，比如数组A中的95,96两个元素前面各填充了3个0；在后面填充2个0，比如数组A中的97,98两个元素后面各填充了2个0
    #（2,3）表示在A的第[1]轴填充（二维数组中，1轴表示列），即在1轴前面填充2个宽度的0，后面填充3个宽度的0
    np.pad(A,((3,2),(2,3)),'constant',constant_values = (0,0))  #constant_values表示填充值，且(before，after)的填充值等于（0,0）
    '''

    def gaus_kernel(winsize, gsigma):
        r = int(winsize / 2)
        c = r
        kernel = np.zeros((winsize, winsize))
        sigma1 = 2 * gsigma * gsigma
        for i in range(-r, r + 1):
            for j in range(-c, c + 1):
                kernel[i + r][j + c] = np.exp(-float(float((i * i + j * j)) / sigma1))
        return kernel

    row, col, depth = image.shape
    # print(row, col)
    sigma2 = 2 * ssigma * ssigma
    gkernel = gaus_kernel(winsize, gsigma)
    # gkernel = copy_self_concatenate(gkernel,depth)

    image_central = image - image[row//2,col//2,:]
    image_central = np.exp(-pow(image_central, 2) / sigma2)
    image_central = np.sum(image_central,2) / depth

    gkernel[image_central > 0.9] = 1
    image_central = image_central * gkernel

    image_central_nband = copy_self_concatenate(image_central,depth)

    # for i in range(r, row - r):
    #     for j in range(c, col - c):
    #         skernel = np.zeros((winsize, winsize))
    #         # print(i, j)
    #         for m in range(-r, r + 1):
    #             for n in range(-c, c + 1):
    #                 # print(m, n)
    #                 # pow()方法返回xy（x的y次方） 的值。
    #                 skernel[m + r][n + c] = np.exp(-pow((image[i][j] - image[i + m][j + n]), 2) / sigma2)
    #
    #                 kernel[m + r][n + c] = skernel[m + r][n + r] * gkernel[m + r][n + r]
    #
    #         sum_kernel = sum(sum(kernel))
    #         kernel = kernel / sum_kernel
    #
    #         for m in range(-r, r + 1):
    #             for n in range(-c, c + 1):
    #
    #                 bilater_image[i][j] = image[i + m][j + n] * kernel[m + r][n + c] + bilater_image[i][j]

    return image_central,image_central_nband
def calculate_theinput_dis(img, coordinate_x, coordinate_y, patch_size):

    sigma2 = 10
    coordinate_x = coordinate_x #- 1
    coordinate_y = coordinate_y #- 1
    window_size = patch_size
    central = window_size // 2
    coordinate_x_begin = coordinate_x - central
    coordinate_x_end = coordinate_x + central + 1
    coordinate_y_begin = coordinate_y - central
    coordinate_y_end = coordinate_y + central + 1
    # print(coordinate_x_begin, coordinate_x_end, coordinate_y_begin, coordinate_y_end)
    print('输入高光谱数据的维度：', img.shape[0], img.shape[1], img.shape[2])

    img1 = img[coordinate_x_begin:coordinate_x_end, coordinate_y_begin:coordinate_y_end, 0:img.shape[2]-1]
    print('输入高光谱数据的维度：', img1.shape[0], img1.shape[1], img1.shape[2])
    skernel = np.zeros(img1.shape)
    # print(skernel.shape)

    for i in range(img1.shape[0]):
        for j in range(img1.shape[1]):
            # print(i, j)
            skernel[i, j, :] = np.exp(-pow((img1[central, central, :] - img1[i, j, :]), 2) / sigma2)

    # print(skernel.shape, skernel[central, central, :])

    skernel = np.sum(skernel, axis=2)
    skernel = abs(skernel - skernel[central, central])
    return skernel


def gaus_kernel(winsize, gsigma):
    r = int(winsize / 2)
    c = r
    kernel = torch.zeros((winsize, winsize))
    sigma1 = 2 * torch.mul(gsigma, gsigma).float()
    if winsize % 2 == 0:
        for i in range(-r, r):
            for j in range(-c, c):
                dis = float(i * i + j * j)
                kernel[i + r][j + c] = torch.exp(- dis / sigma1)
    else:
        for i in range(-r, r + 1):
            for j in range(-c, c + 1):
                dis = float(i * i + j * j)
                kernel[i + r][j + c] = torch.exp(- dis / sigma1)

    return kernel


def copy_self_concatenate(data, depth):
    '''
     二维 重复自己 拼接成三维
    '''

    data = np.reshape(data, (data.shape[0], data.shape[1], 1))
    # print(data.shape, data)

    data1 = data
    while data1.shape[2] < depth:
        data1 = np.concatenate([data1, data], 2)
        # print(data1.shape)
    return data1

def getClosenessWeight1(H, W):
    # 计算空间距离权重模板
    sigma_g = 10
    r, c = np.mgrid[0:H:1, 0:W:1]  # 构造三维表
    r -= int((H - 1) / 2)
    c -= int((W - 1) / 2)
    closeWeight = np.exp(-0.5 * (np.power(r, 2) + np.power(c, 2)) / sigma_g * sigma_g)
    return closeWeight

def jointBLF_double(I, Ig, pr,N1, N2, sigma_d):

    # I 预测的结果图
    # Ig 原始光谱图 引导图
    # pr 概率图 (610, 340, 10)  待处理

    W = N1
    H = N1

    # 构建空间距离权重模板
    closenessWeight = getClosenessWeight1(H, W)

    # 模板的中心点位置
    cH = int((H - 1) / 2)
    cW = int((W - 1) / 2)
    cH1 = int((N2 - 1) / 2)
    cW1 = int((N2 - 1) / 2)

    # 图像矩阵的行数和列数
    # img1_data = pd.DataFrame(I)
    # img1_data2 = pd.ExcelWriter('I.xlsx')  # 写入Excel文件
    # img1_data.to_excel(img1_data2, 'img1', float_format='%.5f')  # ‘page_1’是写入excel的sheet名
    # img1_data2.save()
    # quit()
    # x_begin = PATCH_SIZE // 2
    # x_end   = Ig.shape[0] - x_begin -1
    # y_begin = PATCH_SIZE // 2
    # y_end   = Ig.shape[1] - x_begin -1
    #
    # I = I[x_begin:x_end, y_begin:y_end]
    # Ig = Ig[x_begin:x_end, y_begin:y_end]
    # pr = pr[x_begin:x_end, y_begin:y_end]
    rows, cols = I.shape
    # print(I.shape)
    # img1_data = pd.DataFrame(I)
    # img1_data2 = pd.ExcelWriter('I.xlsx')  # 写入Excel文件
    # img1_data.to_excel(img1_data2, 'img1', float_format='%.5f')  # ‘page_1’是写入excel的sheet名
    # img1_data2.save()
    # quit()

    # 联合双边滤波的结果
    # jblf = np.zeros(I.shape)
    jblf = pr
    # pr = np.pad(pr, ((cH, cH), (cH, cH), (0, 0)), 'constant')
    for r in range(cH, rows - cH, 1):
        for c in range(cW, cols - cW, 1):
            # 当前位置的值
            # r,c = 18,244
            # print(r,c)

            # 当前位置的邻域 N1 邻域
            rTop, rBottom = r - cH, r + cH
            cLeft, cRight = c - cW, c + cW
            # print(rTop, rBottom,cLeft, cRight)

            # 当前位置的邻域 N2 邻域
            rTop1, rBottom1 = r - cH1, r + cH1
            cLeft1, cRight1 = c - cW1, c + cW1
            # print(rTop1, rBottom1, cLeft1, cRight1)

            # 邻域不能超过图像边界
            if rTop1 >= 0 and rBottom1 <= rows and cLeft1 >= 0 and cRight1 <= cols:
                pass
            else:
                if rTop1 < 0:
                    rTop1 = 0
                if rBottom1 > rows:
                    rBottom1 = rows
                if cLeft1 < 0:
                    cLeft1 = 0
                if cRight1 > cols:
                    cRight1 = cols
            # print(rTop1, rBottom1, cLeft1, cRight1)

            # 从 Ig 中截取N2邻域，用于构建相似性权重模板
            region_n2 = Ig[rTop1: rBottom1 + 1, cLeft1: cRight1 + 1,:]
            # print(region_n2.shape)
            d = region_n2 - Ig[r,c,:]
            ddd = 2 * sigma_d * sigma_d
            similarityWeight_n2 = np.exp(- (d*d) / ddd)
            # print(similarityWeight_n2.shape)
            similarityWeight_n2 = np.sum(similarityWeight_n2,2) / region_n2.shape[2]
            # similarityWeight_n2[similarityWeight_n2 > 0.9] = 1
            #
            # img1_data = pd.DataFrame(similarityWeight_n2)
            # img1_data2 = pd.ExcelWriter('similarityWeight2.xlsx')  # 写入Excel文件
            # img1_data.to_excel(img1_data2, 'img1', float_format='%.5f')  # ‘page_1’是写入excel的sheet名
            # img1_data2.save()
            # quit()


            # 从 Ig 中截取N1邻域，用于构建相似性权重模板
            region_n1 = Ig[rTop: rBottom + 1, cLeft: cRight + 1, :]
            # print(region_n1.shape)
            dd = region_n1 - Ig[r,c,:]
            similarityWeight_n1 = np.exp(- (dd*dd) / ddd)
            # print(similarityWeight_n1.shape)
            similarityWeight_n1 = np.sum(similarityWeight_n1,2) / region_n1.shape[2]
            # similarityWeight_n1[similarityWeight_n1 > 0.9] = 1
            # print(similarityWeight_n1)
            # quit()

            pixel = I[r, c]
            # print(pixel)
            region1 = I[rTop1: rBottom1 + 1, cLeft1: cRight1 + 1]
            # print(region1)


            # where_x,where_y = np.where(similarityWeight_n2 == 1)
            # # print(where_x,where_y)
            # where_x = where_x.reshape(len(where_x), 1)
            # where_y = where_y.reshape(len(where_y), 1)
            # where = np.concatenate((where_x,where_y),1)
            # print(where)
            II = region1[similarityWeight_n2 > 0.9]
            g = np.argmax(np.bincount(II))
            # print(g)

            if pixel != g:
                # print('gggggg',pixel,g)
                pr1 = pr[rTop1: rBottom1 + 1, cLeft1: cRight1 + 1,:]
                I2 = pr1[region1 == g]
                # print(I2.shape)
                if I2.shape[0] < N1*N1:
                    # print('fff')
                    weight = closenessWeight * similarityWeight_n1

                    # 将权重归一化
                    weight = weight / np.sum(weight)
                    weight = copy_self_concatenate(weight, pr.shape[2])
                    # print(weight.shape)

                    # 权重模板和邻域对应位置相乘并求和
                    a = pr[rTop:rBottom + 1, cLeft:cRight + 1, :] * weight

                    for i in range(a.shape[2]):
                        # print(a[:,:,i])
                        jblf[r, c, i] = np.sum(a[:, :, i])
                        # print(np.sum(a[:,:,i]))
                        # quit()

                else:
                    I2 = I2[:N1 * N1, :].reshape(N1, N1, -1)
                    # print(I2.shape)

                    s2 = similarityWeight_n2[region1 == g]
                    # print(s2.shape)
                    s2 = s2[:N1 * N1].reshape(N1, N1)
                    # print(s2.shape)

                    # 相似性权重模板和空间距离权重模板相乘
                    weight = closenessWeight * s2

                    # 将权重归一化
                    weight = weight / np.sum(weight)
                    weight = copy_self_concatenate(weight, pr.shape[2])
                    # print(weight.shape)

                    # 权重模板和邻域对应位置相乘并求和
                    a = I2 * weight

                    for i in range(a.shape[2]):
                        # print(a[:,:,i])
                        jblf[r, c, i] = np.sum(a[:, :, i])
                        # print(np.sum(a[:,:,i]))
                        # quit()
            else:
                # 相似性权重模板和空间距离权重模板相乘
                weight = closenessWeight * similarityWeight_n1

                # 将权重归一化
                weight = weight / np.sum(weight)
                weight = copy_self_concatenate(weight,pr.shape[2])
                # print(weight.shape)

                # 权重模板和邻域对应位置相乘并求和
                a = pr[rTop:rBottom + 1, cLeft:cRight + 1, :] * weight

                for i in range(a.shape[2]):
                    # print(a[:,:,i])
                    jblf[r, c, i] = np.sum(a[:,:,i])
                    # print(np.sum(a[:,:,i]))
                    # quit()

                # quit()

                # jblf[r,c,:] = np.sum(pr[rTop:rBottom + 1, cLeft:cRight + 1,:] * weight)

    # jblf = np.pad(jblf, ((x_begin, x_begin + 1), (x_begin, x_begin + 1), (0, 0)), 'constant')
    # print(jblf.shape)
    return jblf
def jointBLF_double1(I, Ig, pr,N1, N2, sigma_d):

    # I 预测的结果图
    # Ig 原始光谱图 引导图
    # pr 概率图 (610, 340, 10)  待处理


    W = N1
    H = N1

    # 构建空间距离权重模板
    closenessWeight = getClosenessWeight1(H, W)

    # 模板的中心点位置
    cH = int((H - 1) / 2)
    cW = int((W - 1) / 2)
    cH1 = int((N2 - 1) / 2)
    cW1 = int((N2 - 1) / 2)

    # 图像矩阵的行数和列数
    # img1_data = pd.DataFrame(I)
    # img1_data2 = pd.ExcelWriter('I.xlsx')  # 写入Excel文件
    # img1_data.to_excel(img1_data2, 'img1', float_format='%.5f')  # ‘page_1’是写入excel的sheet名
    # img1_data2.save()
    # quit()
    # x_begin = PATCH_SIZE // 2
    # x_end   = Ig.shape[0] - x_begin -1
    # y_begin = PATCH_SIZE // 2
    # y_end   = Ig.shape[1] - x_begin -1
    #
    # I = I[x_begin:x_end, y_begin:y_end]
    # Ig = Ig[x_begin:x_end, y_begin:y_end]
    # pr = pr[x_begin:x_end, y_begin:y_end]
    rows, cols = I.shape
    # print(I.shape)
    # img1_data = pd.DataFrame(I)
    # img1_data2 = pd.ExcelWriter('I.xlsx')  # 写入Excel文件
    # img1_data.to_excel(img1_data2, 'img1', float_format='%.5f')  # ‘page_1’是写入excel的sheet名
    # img1_data2.save()
    # quit()

    # 联合双边滤波的结果
    # jblf = np.zeros(I.shape)
    # jblf = np.zeros(pr.shape)
    jblf = pr
    # pr = np.pad(pr, ((cH, cH), (cH, cH), (0, 0)), 'edge')
    for r in range(cH, rows - cH, 1):
        for c in range(cW, cols - cW, 1):
            # if gt1[r,c] != 0:
            # 当前位置的值
            # r,c = 138,127
            # print(r,c)

            # 当前位置的邻域 N1 邻域
            rTop, rBottom = r - cH, r + cH
            cLeft, cRight = c - cW, c + cW
            # print(rTop, rBottom,cLeft, cRight)

            # 当前位置的邻域 N2 邻域
            rTop1, rBottom1 = r - cH1, r + cH1
            cLeft1, cRight1 = c - cW1, c + cW1
            # print(rTop1, rBottom1, cLeft1, cRight1)

            # 邻域不能超过图像边界
            if rTop1 >= 0 and rBottom1 <= rows and cLeft1 >= 0 and cRight1 <= cols:
                pass
            else:
                if rTop1 < 0:
                    rTop1 = 0
                if rBottom1 > rows:
                    rBottom1 = rows
                if cLeft1 < 0:
                    cLeft1 = 0
                if cRight1 > cols:
                    cRight1 = cols
            # print(rTop1, rBottom1, cLeft1, cRight1)

            # 从 Ig 中截取N2邻域，用于构建相似性权重模板
            region_n2 = Ig[rTop1: rBottom1 + 1, cLeft1: cRight1 + 1, :]
            # print(region_n2.shape)
            d = region_n2 - Ig[r, c, :]
            ddd = 2 * sigma_d * sigma_d
            similarityWeight_n2 = np.exp(- (d * d) / ddd)
            # print(similarityWeight_n2.shape)
            similarityWeight_n2 = np.sum(similarityWeight_n2, 2) / region_n2.shape[2]
            similarityWeight_n2[similarityWeight_n2 > 0.97] = 1
            #
            # img1_data = pd.DataFrame(similarityWeight_n2)
            # img1_data2 = pd.ExcelWriter('similarityWeight2.xlsx')  # 写入Excel文件
            # img1_data.to_excel(img1_data2, 'img1', float_format='%.5f')  # ‘page_1’是写入excel的sheet名
            # img1_data2.save()
            # quit()

            # 从 Ig 中截取N1邻域，用于构建相似性权重模板
            region_n1 = Ig[rTop: rBottom + 1, cLeft: cRight + 1, :]
            # print(region_n1.shape)
            dd = region_n1 - Ig[r, c, :]
            similarityWeight_n1 = np.exp(- (dd * dd) / ddd)
            # print(similarityWeight_n1.shape)
            similarityWeight_n1 = np.sum(similarityWeight_n1, 2) / region_n1.shape[2]
            # similarityWeight_n1[similarityWeight_n1 > 0.9] = 1
            # print(similarityWeight_n1)
            # quit()

            pixel = I[r, c]
            # print(pixel)
            region1 = I[rTop1: rBottom1 + 1, cLeft1: cRight1 + 1]
            # print(region1)

            # where_x,where_y = np.where(similarityWeight_n2 == 1)
            # # print(where_x,where_y)
            # where_x = where_x.reshape(len(where_x), 1)
            # where_y = where_y.reshape(len(where_y), 1)
            # where = np.concatenate((where_x,where_y),1)
            # print(where)
            II = region1[similarityWeight_n2 > 0.97]  #  for splbf
            # II = region1[similarityWeight_n2 > 0.9]  #   for dwlbf
            g = np.argmax(np.bincount(II))
            # print(g)

            if pixel != g:
                # print('gggggg',pixel,g)
                pr1 = pr[rTop1: rBottom1 + 1, cLeft1: cRight1 + 1, :]
                I2 = pr1[region1 == g]
                # print(I2.shape)
                if I2.shape[0] < N1 * N1:
                    # print('fff')
                    weight = closenessWeight * similarityWeight_n1

                    # 将权重归一化
                    weight = weight / np.sum(weight)
                    weight = copy_self_concatenate(weight, pr.shape[2])
                    # print(weight.shape)

                    # 权重模板和邻域对应位置相乘并求和
                    a = pr[rTop:rBottom + 1, cLeft:cRight + 1, :] * weight

                    for i in range(a.shape[2]):
                        # print(a[:,:,i])
                        jblf[r, c, i] = np.sum(a[:, :, i])
                        # print(np.sum(a[:,:,i]))
                        # quit()

                else:
                    I2 = I2[:N1 * N1, :].reshape(N1, N1, -1)
                    # print(I2.shape)

                    s2 = similarityWeight_n2[region1 == g]
                    # print(s2.shape)
                    s2 = s2[:N1 * N1].reshape(N1, N1)
                    # print(s2.shape)

                    # 相似性权重模板和空间距离权重模板相乘
                    weight = closenessWeight * s2

                    # 将权重归一化
                    weight = weight / np.sum(weight)
                    weight = copy_self_concatenate(weight, pr.shape[2])
                    # print(weight.shape)

                    # 权重模板和邻域对应位置相乘并求和
                    a = I2 * weight
                    aa = a.sum((0, 1))
                    jblf[r, c, :] = aa

                    # for i in range(a.shape[2]):
                    #     # print(a[:,:,i])
                    #     jblf[r, c, i] = np.sum(a[:, :, i])
                    #     # print(np.sum(a[:,:,i]))
                    #     # quit()
            # else:
            #     # 相似性权重模板和空间距离权重模板相乘
            #     weight = closenessWeight * similarityWeight_n1
            #
            #     # 将权重归一化
            #     weight = weight / np.sum(weight)
            #     weight = copy_self_concatenate(weight, pr.shape[2])
            #     # print(weight.shape)
            #
            #     # 权重模板和邻域对应位置相乘并求和
            #     a = pr[rTop:rBottom + 1, cLeft:cRight + 1, :] * weight
            #
            #     for i in range(a.shape[2]):
            #         # print(a[:,:,i])
            #         jblf[r, c, i] = np.sum(a[:, :, i])
            #         # print(np.sum(a[:,:,i]))
            #         # quit()

                # quit()

                # jblf[r,c,:] = np.sum(pr[rTop:rBottom + 1, cLeft:cRight + 1,:] * weight)

    # jblf = np.pad(jblf, ((x_begin, x_begin + 1), (x_begin, x_begin + 1), (0, 0)), 'constant')
    # print(jblf.shape)
    return jblf

def jointBLF_double1_for_pixel(prediction, img, probabilities, closenessWeight, r, c):

    # prediction 预测的结果图
    # img 原始光谱图 引导图
    # probabilities 概率图 (610, 340, 10)  待处理

    N1, N2 = 3, 25
    sigma_d = 0.1
    W = N1
    H = N1

    # 模板的中心点位置
    cH = int((H - 1) / 2)
    cW = int((W - 1) / 2)
    cH1 = int((N2 - 1) / 2)
    cW1 = int((N2 - 1) / 2)

    # 图像矩阵的行数和列数
    # img1_data = pd.DataFrame(I)
    # img1_data2 = pd.ExcelWriter('I.xlsx')  # 写入Excel文件
    # img1_data.to_excel(img1_data2, 'img1', float_format='%.5f')  # ‘page_1’是写入excel的sheet名
    # img1_data2.save()
    # quit()
    # x_begin = PATCH_SIZE // 2
    # x_end   = Ig.shape[0] - x_begin -1
    # y_begin = PATCH_SIZE // 2
    # y_end   = Ig.shape[1] - x_begin -1
    #
    # I = I[x_begin:x_end, y_begin:y_end]
    # Ig = Ig[x_begin:x_end, y_begin:y_end]
    # pr = pr[x_begin:x_end, y_begin:y_end]
    rows, cols = prediction.shape
    # print(I.shape)
    # img1_data = pd.DataFrame(I)
    # img1_data2 = pd.ExcelWriter('I.xlsx')  # 写入Excel文件
    # img1_data.to_excel(img1_data2, 'img1', float_format='%.5f')  # ‘page_1’是写入excel的sheet名
    # img1_data2.save()
    # quit()

    # 联合双边滤波的结果
    # jblf = np.zeros(I.shape)
    # jblf = np.zeros(pr.shape)
    jblf = probabilities
    # pr = np.pad(pr, ((cH, cH), (cH, cH), (0, 0)), 'edge')
    # 当前位置的值
    # r, c = 18, 244
    print(r, c)

    # 当前位置的邻域 N1 邻域
    rTop, rBottom = r - cH, r + cH
    cLeft, cRight = c - cW, c + cW
    # print(rTop, rBottom,cLeft, cRight)

    # 当前位置的邻域 N2 邻域
    rTop1, rBottom1 = r - cH1, r + cH1
    cLeft1, cRight1 = c - cW1, c + cW1
    # print(rTop1, rBottom1, cLeft1, cRight1)

    # 邻域不能超过图像边界
    if rTop1 >= 0 and rBottom1 <= rows and cLeft1 >= 0 and cRight1 <= cols:
        pass
    else:
        if rTop1 < 0:
            rTop1 = 0
        if rBottom1 > rows:
            rBottom1 = rows
        if cLeft1 < 0:
            cLeft1 = 0
        if cRight1 > cols:
            cRight1 = cols
    # print(rTop1, rBottom1, cLeft1, cRight1)

    # 从 Ig 中截取N2邻域，用于构建相似性权重模板
    region_n2 = img[rTop1: rBottom1 + 1, cLeft1: cRight1 + 1, :]
    # print(region_n2.shape)
    d = region_n2 - img[r, c, :]
    ddd = 2 * sigma_d * sigma_d
    similarityWeight_n2 = np.exp(- (d * d) / ddd)
    # print(similarityWeight_n2.shape)
    similarityWeight_n2 = np.sum(similarityWeight_n2, 2) / region_n2.shape[2]
    # similarityWeight_n2[similarityWeight_n2 > 0.9] = 1
    #
    # img1_data = pd.DataFrame(similarityWeight_n2)
    # img1_data2 = pd.ExcelWriter('similarityWeight2.xlsx')  # 写入Excel文件
    # img1_data.to_excel(img1_data2, 'img1', float_format='%.5f')  # ‘page_1’是写入excel的sheet名
    # img1_data2.save()
    # quit()

    # 从 Ig 中截取N1邻域，用于构建相似性权重模板
    region_n1 = img[rTop: rBottom + 1, cLeft: cRight + 1, :]
    # print(region_n1.shape)
    dd = region_n1 - img[r, c, :]
    similarityWeight_n1 = np.exp(- (dd * dd) / ddd)
    # print(similarityWeight_n1.shape)
    similarityWeight_n1 = np.sum(similarityWeight_n1, 2) / region_n1.shape[2]
    # similarityWeight_n1[similarityWeight_n1 > 0.9] = 1
    # print(similarityWeight_n1)
    # quit()

    pixel = prediction[r, c]
    # print(pixel)
    region1 = prediction[rTop1: rBottom1 + 1, cLeft1: cRight1 + 1]
    # print(region1)

    # where_x,where_y = np.where(similarityWeight_n2 == 1)
    # # print(where_x,where_y)
    # where_x = where_x.reshape(len(where_x), 1)
    # where_y = where_y.reshape(len(where_y), 1)
    # where = np.concatenate((where_x,where_y),1)
    # print(where)
    II = region1[similarityWeight_n2 > 0.9]
    g = np.argmax(np.bincount(II))
    # print(g)

    if pixel != g:

        pr1 = probabilities[rTop1: rBottom1 + 1, cLeft1: cRight1 + 1, :]
        I2 = pr1[region1 == g]
        # print(I2.shape)
        if I2.shape[0] < N1 * N1:
            # print('fff')
            weight = closenessWeight * similarityWeight_n1

            # 将权重归一化
            weight = weight / np.sum(weight)
            weight = copy_self_concatenate(weight, probabilities.shape[2])
            # print(weight.shape)

            # 权重模板和邻域对应位置相乘并求和
            a = probabilities[rTop:rBottom + 1, cLeft:cRight + 1, :] * weight

            aa = a.sum((0, 1))
            jblf[r, c, :] = aa
        else:
            I2 = I2[:N1 * N1, :].reshape(N1, N1, -1)
            # print(I2.shape)

            s2 = similarityWeight_n2[region1 == g]
            # print(s2.shape)
            s2 = s2[:N1 * N1].reshape(N1, N1)
            # print(s2.shape)

            # 相似性权重模板和空间距离权重模板相乘
            weight = closenessWeight * s2

            # 将权重归一化
            weight = weight / np.sum(weight)
            weight = copy_self_concatenate(weight, probabilities.shape[2])
            # print(weight.shape)

            # 权重模板和邻域对应位置相乘并求和
            a = I2 * weight
            aa = a.sum((0, 1))
            jblf[r, c, :] = aa
    return jblf

def superpixel_jointBLF(I, Ig, pr,N1, N2, sigma_d, segments):

    # I 预测的结果图
    # Ig 原始光谱图 引导图
    # pr 概率图 (610, 340, 10)  待处理
    # segments 超像素分割结果图

    # img1_data = pd.DataFrame(segments)
    # img1_data2 = pd.ExcelWriter('segments.xlsx')  # 写入Excel文件
    # img1_data.to_excel(img1_data2, 'segments', float_format='%.5f')  # ‘page_1’是写入excel的sheet名
    # img1_data2.save()
    # quit()

    # 构建空间距离权重模板
    closenessWeight = getClosenessWeight1(N1, N1)
    pixel_num = np.unique(segments)
    jblf = pr.copy()
    for i in pixel_num:
        print(i)

        Ig_mask = Ig.copy()
        I_mask = I.copy()

        x, y = np.where(segments == i)

        I_mask[x, y] = 0
        I_mask = I - I_mask

        class_prediction_one = I[x, y]
        class_prediction_num = np.unique(class_prediction_one)

        class_prediction_one_most = np.argmax(np.bincount(class_prediction_one))

        if len(class_prediction_num) > 1:

            x_most, y_most = np.where(I_mask == class_prediction_one_most)

            i2 = pr[x_most, y_most]

            ig_one = Ig[x_most, y_most]
            ig_one_mean  = np.mean(ig_one)

            for j in class_prediction_num:

                if j != class_prediction_one_most:

                    x_1, y_1 = np.where(I_mask == j)

                    Ig_mask[x_1, y_1] = 0
                    Ig_mask1 = Ig - Ig_mask

                    Ig_mask2 = Ig_mask1.copy()
                    Ig_mask2[np.where(Ig_mask1 != 0)] = ig_one_mean

                    img_ = Ig_mask2 - Ig_mask1

                    c = np.exp(-(img_ * img_) / 0.02)


                    cc = c.sum(2) / c.shape[2]

                    cc[cc == 1] = 0
                    m, n = np.where(cc > 0.5)
                    if len(m) > 0:
                        if len(i2) >= N1 * N1:
                            i2 = i2[:N1 * N1, :].reshape(N1, N1, -1)
                            s2 = cc[m, n]
                            if len(s2) < N1 * N1:
                                s2 = np.append(s2, s2)
                                s2 = np.append(s2, s2)
                                s2 = np.append(s2, s2)
                                s2 = np.append(s2, s2)

                            s2 = s2[:N1 * N1].reshape(N1, N1)

                            # 相似性权重模板和空间距离权重模板相乘
                            weight = closenessWeight * s2

                            # 将权重归一化
                            weight = weight / np.sum(weight)
                            weight = copy_self_concatenate(weight, pr.shape[2])
                            # print(weight.shape)

                            # 权重模板和邻域对应位置相乘并求和
                            a = i2 * weight
                            aa = a.sum((0, 1))

                            jblf[m, n] = aa


    return jblf
def superpixel_jointBLF1(prediction, img, probabilities,N1, N2, sigma_d, segments):

    # prediction 预测的结果图
    # img 原始光谱图 引导图
    # probabilities 概率图 (610, 340, 10)  待处理
    # segments 超像素分割结果图

    # img1_data = pd.DataFrame(segments)
    # img1_data2 = pd.ExcelWriter('segments.xlsx')  # 写入Excel文件
    # img1_data.to_excel(img1_data2, 'segments', float_format='%.5f')  # ‘page_1’是写入excel的sheet名
    # img1_data2.save()
    # quit()

    # 构建空间距离权重模板
    rows, cols = prediction.shape

    closenessWeight = getClosenessWeight1(N1, N1)
    pixel_num = np.unique(segments)
    probabilities_lbf = probabilities.copy()
    img_mask = img.copy()
    mask = np.ones(prediction.shape, dtype='bool')

    for r in range(0, rows, 1):
        for c in range(0, cols, 1):

            if mask[r, c] == 1:

                prediction_mask = prediction.copy()
                print(r, c)
                # r, c = 70, 285

                seg_one = segments[r, c]
                x, y = np.where(segments == seg_one)


                prediction_mask[x, y] = 0
                prediction_mask = prediction - prediction_mask

                class_prediction_one = prediction[x, y]
                class_prediction_num = np.unique(class_prediction_one)
                class_prediction_one_most = np.argmax(np.bincount(class_prediction_one))
                x_most, y_most = np.where(prediction_mask == class_prediction_one_most)

                if len(class_prediction_num) == 1 or len(x_most) < N1 * N1:
                    print('代表该超像素块预测结果一致，不需要滤波')
                    mask[x, y] = 0

                else:
                    print('代表该超像素块预测结果不一致，需要滤波')

                    x_not_most = []
                    y_not_most = []
                    for k in range(len(class_prediction_num)):
                        num = class_prediction_num[k]
                        if num != class_prediction_one_most:
                            x_not, y_not = np.where(prediction_mask == num)
                            x_not_most.extend(x_not)
                            y_not_most.extend(y_not)

                    # class_prediction_num
                    img_most = img_mask[x_most, y_most]
                    probabilities_most = probabilities[x_most, y_most]
                    for m in range(len(x_not_most)):
                        # print(len(x_not_most))
                        # print('begin to filt each pixel')

                        x_one, y_one = x_not_most[m], y_not_most[m]
                        img_one = img_mask[x_one, y_one]
                        img_cal = img_one - img_most

                        similarityWeight = np.exp(- (img_cal * img_cal) / 2 * sigma_d * sigma_d)
                        # print(similarityWeight.shape)

                        similarityWeight = np.sum(similarityWeight, 1) / img_cal.shape[1]

                        i2 = probabilities_most[:N1 * N1, :].reshape(N1, N1, -1)
                        # print(i2.shape)

                        similarityWeight = similarityWeight[:N1 * N1].reshape(N1, N1)
                        # 相似性权重模板和空间距离权重模板相乘
                        weight = closenessWeight * similarityWeight
                        # print(weight.shape)

                        # 将权重归一化
                        weight = weight / np.sum(weight)
                        weight = copy_self_concatenate(weight, probabilities.shape[2])
                        # print(weight.shape)
                        # 权重模板和邻域对应位置相乘并求和
                        a = i2 * weight
                        # print(a.shape)
                        aa = a.sum((0, 1))
                        # print(aa.shape)
                        probabilities_lbf[x_one, y_one] = aa

                    mask[x, y] = 0
    return probabilities_lbf
def superpixel_jointBLF2(prediction, img, probabilities,N1, N2, ddd, segments):

    # prediction 预测的结果图
    # img 原始光谱图 引导图
    # probabilities 概率图 (610, 340, 10)  待处理
    # segments 超像素分割结果图

    # img1_data = pd.DataFrame(segments)
    # img1_data2 = pd.ExcelWriter('segments.xlsx')  # 写入Excel文件
    # img1_data.to_excel(img1_data2, 'segments', float_format='%.5f')  # ‘page_1’是写入excel的sheet名
    # img1_data2.save()
    # quit()

    # 构建空间距离权重模板

    W = N1
    H = N1
    # 模板的中心点位置
    cH = int((H - 1) / 2)
    cW = int((W - 1) / 2)
    cH1 = int((N2 - 1) / 2)
    cW1 = int((N2 - 1) / 2)
    rows, cols = prediction.shape

    closenessWeight = getClosenessWeight1(N1, N1)
    # pixel_num = np.unique(segments)
    probabilities_lbf = probabilities.copy()
    # img = img[:,:,:img.shape[2]]
    img_mask = img.copy()
    mask = np.ones(prediction.shape, dtype='bool')

    def dwlbf_for_one_pixel(probabilities, qq, pp):
        # qq, pp = rows - 1, cols - 1
        # 当前位置的邻域 N1 邻域
        rTop, rBottom = qq - cH, qq + cH
        cLeft, cRight = pp - cW, pp + cW

        # 当前位置的邻域 N2 邻域
        rTop1, rBottom1 = qq - cH1, qq + cH1
        cLeft1, cRight1 = pp - cW1, pp + cW1
        # print(qq, pp, rTop, rBottom , cLeft, cRight )
        # print(qq, pp, rTop1, rBottom1, cLeft1, cRight1)

        # 邻域不能超过图像边界
        if rTop1 < 0:
            rTop1 = 0
            rBottom1 = rBottom1 + cH1
        if rBottom1 >= rows:
            rBottom1 = rows - 1
            rTop1 = rTop1 - cH1
        if cLeft1 < 0:
            cLeft1 = 0
            cRight1 = cRight1 + cW1
        if cRight1 >= cols:
            cRight1 = cols - 1
            cLeft1 = cLeft1 - cW1

        if rTop < 0:
            rTop = 0
            rBottom = rBottom + cH
        if rBottom >= rows:
            rBottom = rows - 1
            rTop = rTop - cH
        if cLeft < 0:
            cLeft = 0
            cRight = cRight + cW
        if cRight >= cols:
            cRight = cols - 1
            cLeft = cLeft - cW
        # print(qq, pp, rTop, rBottom , cLeft, cRight )
        # print(qq, pp, rTop1, rBottom1, cLeft1, cRight1)

        # 从 Ig 中截取N2邻域，用于构建相似性权重模板
        region_n2 = img[rTop1: rBottom1 + 1, cLeft1: cRight1 + 1, :]
        d = region_n2 - img[qq, pp, :]
        similarityWeight_n2 = np.exp(- (d * d) / ddd)
        similarityWeight_n2 = np.sum(similarityWeight_n2, 2) / region_n2.shape[2]

        # 从 Ig 中截取N1邻域，用于构建相似性权重模板
        region_n1 = img[rTop: rBottom + 1, cLeft: cRight + 1, :]
        # region_n1 = img[608: 610 + 1, 338: 340 + 1, :]
        # print(region_n1.shape, region_n2.shape)
        # if region_n1.shape[0] != 3 or region_n1.shape[1] != 3:
        #     print(region_n1.shape, region_n2.shape)
        #     print(qq, pp, rTop, rBottom + 1, cLeft, cRight + 1)
        #     print(qq, pp, rTop1, rBottom1 + 1, cLeft1, cRight1 + 1)
        #     quit()

        dd = region_n1 - img[qq, pp, :]
        similarityWeight_n1 = np.exp(- (dd * dd) / ddd)
        similarityWeight_n1 = np.sum(similarityWeight_n1, 2) / region_n1.shape[2]

        pixel = prediction[qq, pp]
        region1 = prediction[rTop1: rBottom1 + 1, cLeft1: cRight1 + 1]

        II = region1[similarityWeight_n2 > 0.97]
        g = np.argmax(np.bincount(II))
        # print(pixel, qq, pp, rTop,rBottom + 1, cLeft,cRight + 1 ,g)

        # img1_data = pd.DataFrame(prediction)
        # img1_data2 = pd.ExcelWriter('prediction.xlsx')  # 写入Excel文件
        # img1_data.to_excel(img1_data2, 'prediction', float_format='%.5f')  # ‘page_1’是写入excel的sheet名
        # img1_data2.save()
        # quit()

        if pixel != g:

            pr1 = probabilities[rTop1: rBottom1 + 1, cLeft1: cRight1 + 1, :]
            I2 = pr1[region1 == g]
            if I2.shape[0] < N1 * N1:

                weight = closenessWeight * similarityWeight_n1

                # 将权重归一化
                weight = weight / np.sum(weight)
                weight = copy_self_concatenate(weight, probabilities.shape[2])

                # 权重模板和邻域对应位置相乘并求和
                a = probabilities[rTop:rBottom + 1, cLeft:cRight + 1, :] * weight
                aa = a.sum((0, 1))
                probabilities_lbf[qq, pp] = aa
            else:
                I2 = I2[:N1 * N1, :].reshape(N1, N1, -1)
                s2 = similarityWeight_n2[region1 == g]
                s2 = s2[:N1 * N1].reshape(N1, N1)

                # 相似性权重模板和空间距离权重模板相乘
                weight = closenessWeight * s2

                # 将权重归一化
                weight = weight / np.sum(weight)
                weight = copy_self_concatenate(weight, probabilities.shape[2])

                # 权重模板和邻域对应位置相乘并求和
                a = I2 * weight
                aa = a.sum((0, 1))
                probabilities_lbf[qq, pp] = aa
    for r in range(cH, rows - cH, 1):
        for c in range(cW, cols - cW, 1):
            # r, c = 302, 112
            # r, c = 166, 4
            if mask[r, c] == 1:

                prediction_mask = prediction.copy()
                seg_one = segments[r, c]

                x, y = np.where(segments == seg_one)

                prediction_mask[x, y] = 0
                prediction_mask = prediction - prediction_mask

                class_prediction_one = prediction[x, y]
                class_prediction_num = np.unique(class_prediction_one)
                # print(class_prediction_num)

                if len(class_prediction_num) == 1:
                    # print('代表该超像素块预测结果一致，不需要SP滤波, 进行 DWLBF 滤波',r,c)
                    for kk in range(len(x)):

                        dwlbf_for_one_pixel(probabilities,x[kk], y[kk])

                elif len(class_prediction_num) > 1:
                    # print('代表该超像素块预测结果不一致，需要 SP 滤波,然后再进行 DWLBF 滤波',r,c)

                    class_prediction_one_most = np.argmax(np.bincount(class_prediction_one))
                    # print(class_prediction_one_most)

                    x_most, y_most = np.where(prediction_mask == class_prediction_one_most)
                    # print(x_most, y_most)
                    img_one_most = img[x_most, y_most]
                    img_one_most = np.mean(img_one_most, axis=0)
                    x_not_most = []
                    y_not_most = []
                    for k in range(len(class_prediction_num)):
                        num = class_prediction_num[k]
                        if num != class_prediction_one_most:
                            x_not, y_not = np.where(prediction_mask == num)
                            # print('hhhhhh',x_not, y_not)
                            #
                            # img_one_not = img[x_not, y_not]
                            # img_one_not = np.mean(img_one_not, axis=0)
                            # img_cal = img_one_not - img_one_most
                            #
                            # sim = np.exp(- (img_cal * img_cal) / 0.001)
                            # sim_sum = sim.sum() / img.shape[2]
                            # print(sim_sum)
                            # # quit()

                            # if sim > 0.95:
                            #
                            x_not_most.extend(x_not)
                            y_not_most.extend(y_not)

                    # quit()

                    if len(x_not_most) > len(x_most):
                        probabilities_lbf[x_not_most[:len(x_most)], y_not_most[:len(y_most)]] = probabilities_lbf[x_most, y_most]
                        prediction[x_not_most[:len(x_most)], y_not_most[:len(y_most)]] = prediction[
                            x_most, y_most]
                    else:
                        probabilities_lbf[x_not_most, y_not_most] = probabilities_lbf[
                                x_most[:len(x_not_most)], y_most[:len(y_not_most)]]
                        prediction[x_not_most, y_not_most] = prediction[
                            x_most[:len(x_not_most)], y_most[:len(y_not_most)]]

                    for kk in range(len(x)):
                        # print(x[kk], y[kk]),
                        # dwlbf_for_one_pixel(probabilities, 166, 3)
                        dwlbf_for_one_pixel(probabilities, x[kk], y[kk])

                mask[x, y] = 0

    return probabilities_lbf


def calculate_the_dis(which_class, train_gt):

        coo = np.argwhere(train_gt == which_class)
        skernel1 = np.zeros((1, len(coo) - 1))

        for i in range(0, len(coo) - 1):
            skernel1[:, i] = np.exp(
                -pow((img[coo[0][0], coo[0][1], :] - img[coo[i + 1][0], coo[i + 1][1], :]), 2) / 10).sum(axis=0)

        return np.mean(skernel1) - img.shape[2]

def superpixel_filter(prediction, Seg):

        pixel_num = np.unique(Seg)
        for i in pixel_num:
            x, y = np.where(Seg == i)

            prediction_one = prediction[x, y]
            ccc = np.argmax(np.bincount(prediction_one))

            if ccc != np.mean(prediction_one):
                # print('need to be firlter')
                prediction[x, y] = ccc

        return prediction


