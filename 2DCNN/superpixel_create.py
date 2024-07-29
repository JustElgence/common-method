import numpy as np

from skimage.segmentation import slic, mark_boundaries, felzenszwalb, quickshift, random_walker

import cv2
import math


def LSC_superpixel(I, nseg):
    superpixelNum = nseg
    ratio = 0.075
    size = int(math.sqrt(((I.shape[0] * I.shape[1]) / nseg)))
    superpixelLSC = cv2.ximgproc.createSuperpixelLSC(
        I,
        region_size=size,
        ratio=0.005)
    superpixelLSC.iterate()
    superpixelLSC.enforceLabelConnectivity(min_element_size=25)
    segments = superpixelLSC.getLabels()
    return np.array(segments, np.int64)


def SEEDS_superpixel(I, nseg):
    I = np.array(I[:, :, 0:3], np.float32).copy()
    I_new = cv2.cvtColor(I, cv2.COLOR_BGR2HSV)
    # I_new =np.array( I[:,:,0:3],np.float32).copy()
    height, width, channels = I_new.shape

    superpixelNum = nseg
    seeds = cv2.ximgproc.createSuperpixelSEEDS(width, height, channels, int(superpixelNum), num_levels=2, prior=1,
                                               histogram_bins=5)
    seeds.iterate(I_new, 4)
    segments = seeds.getLabels()
    # segments=SegmentsLabelProcess(segments) # 排除labels中不连续的情况
    return segments


def SegmentsLabelProcess(labels):
    '''
    对labels做后处理，防止出现label不连续现象
    '''
    labels = np.array(labels, np.int64)
    H, W = labels.shape
    ls = list(set(np.reshape(labels, [-1]).tolist()))

    dic = {}
    for i in range(len(ls)):
        dic[ls[i]] = i

    new_labels = labels
    for i in range(H):
        for j in range(W):
            new_labels[i, j] = dic[new_labels[i, j]]
    return new_labels

class SLIC(object):
    def __init__(self, HSI, labels, n_segments=1000, compactness=20, max_iter=10, sigma=0, min_size_factor=0.3,
                 max_size_factor=3):
        self.n_segments = n_segments
        self.compactness = compactness
        self.max_iter = max_iter
        self.min_size_factor = min_size_factor
        self.max_size_factor = max_size_factor
        self.sigma = sigma
        self.data = HSI

    def get_Segments(self):
        # 执行 SLCI 并得到Q(nxm),S(m*b)
        img = self.data
        (h, w, d) = img.shape
        print(img.shape)
        # 计算超像素S以及相关系数矩阵Q
        segments = slic(img, n_segments=self.n_segments, compactness=self.compactness, max_iter=self.max_iter,
                        convert2lab=False, sigma=self.sigma, enforce_connectivity=True,
                        min_size_factor=self.min_size_factor, max_size_factor=self.max_size_factor, slic_zero=False)

        # train_gt_data = pd.DataFrame(segments)
        # train_gt_data2 = pd.ExcelWriter('segments.xlsx')  # 写入Excel文件
        # train_gt_data.to_excel(train_gt_data2, 'segments', float_format='%.5f')  # ‘page_1’是写入excel的sheet名
        # train_gt_data2.save()
        # quit()

        # segments = felzenszwalb(img, scale=1,sigma=0.5,min_size=25)
        #
        # segments = quickshift(img.astype('double'),ratio=1,kernel_size=5,max_dist=4,sigma=0.8, convert2lab=False)
        # #
        # segments=LSC_superpixel(img,self.n_segments)
        #
        # segments=SEEDS_superpixel(img,self.n_segments)

        # 判断超像素label是否连续,否则予以校正
        if segments.max() + 1 != len(list(set(np.reshape(segments, [-1]).tolist()))): segments = SegmentsLabelProcess(
            segments)

        self.segments = segments

        return self.segments

class LDA_SLIC(object):
    def __init__(self, img, data, labels, n_component):
        self.data = data
        self.init_labels = labels
        self.curr_data = data
        self.n_component = n_component
        self.height, self.width, self.bands = img.shape
        self.x_flatt = np.reshape(img, [self.width * self.height, self.bands])
        self.y_flatt = np.reshape(labels, [self.height * self.width])
        self.labes = labels

    def SLIC_Process(self, img1, scale=25):
        n_segments_init = self.height * self.width / scale
        print("n_segments_init", n_segments_init)
        myslic = SLIC(img1, n_segments=n_segments_init, labels=self.labes, compactness=1, max_iter=10, sigma=0,
                      min_size_factor=0.1, max_size_factor=3, )

        Segments = myslic.get_Segments()

        return Segments


    def simple_superpixel_no_LDA(self, scale):
        Seg = self.SLIC_Process(self.data, scale=scale)
        # Seg = self.SLIC_Process(self.data, scale=scale)
        return Seg