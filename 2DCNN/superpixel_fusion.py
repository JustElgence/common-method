
import torch
import torch.utils.data as data
from torchsummary import summary
from sklearn.metrics.pairwise import euclidean_distances
import scipy.io as io
import numpy as np
import pandas as pd
from datasets import get_dataset
from utils import pca_img


def smooth_pixel(img, A, seg100,  threshold):

    Segments100_out = np.zeros([seg100.shape[0], seg100.shape[1]])
    # dd = seg100.reshape(-1,1).tolist()
    # # print(dd.shape)
    # dd = set(dd)
    # print(dd)
    # quit()
    pixel_num = np.arange(0, np.max(seg100))

    for i in range(np.max(seg100) + 1):
        # i = 1390
        if i in pixel_num:
            # print(i)
            num = np.where(A[i, :] == 1)   # 和 i 相邻的 超像素 编号
            # print(num[0])
            # b = np.zeros([1 + len(num[0]), img.shape[2]])
            bb = img[seg100 == i]   # 超像素编号 i 的位置的 img 的光谱向量均值
            # print(bb.shape)
            bb = np.mean(bb, axis=0)
            # print(bb.shape)

            for j in range(len(num[0])):

                img_one = img[seg100 == num[0][j]]  # # 超像素编号 num[0][j] 的位置的 img 的光谱向量均值， 就是 i 的邻域
                img_one = np.mean(img_one, axis=0)
                # c = euclidean_distances(bb, img_one)
                c = bb - img_one   # 求光谱相似性，高斯核函数
                c = np.exp(-(c * c) / 0.005)
                cc = c.sum() / len(bb)

                # cc = np.sum(c) / c.shape[1] / c.shape[0]
                # print(cc)

                # 大于 阈值 ，判定为 同一类， 然后将 num[0][j] 编号 变成跟 i ，即合并超像素块
                if cc >= threshold:
                    seg100[seg100 == num[0][j]] = i
                    index = np.where(pixel_num == num[0][j])
                    pixel_num = np.delete(pixel_num, index)
                    A[:, num[0][j]] = 0
        # quit()

    name = 'Segments100_out.mat'
    io.savemat(name, {'Segments': seg100})


def smooth_pixel_one(img, A, seg100,  threshold):

    # Segments100_out = np.zeros([seg100.shape[0], seg100.shape[1]])
    # dd = seg100.reshape(-1,1).tolist()
    # # print(dd.shape)
    # dd = set(dd)
    # print(dd)
    # quit()
    # pixel_num = np.arange(0, np.max(seg100))
    # print(pixel_num)
    pixel_num = np.unique(seg100)

    for i in range(np.max(pixel_num) + 1):
        # i = 1390
        if i in pixel_num:
            # print(i)
            num = np.where(A[i, :] == 1)   # 和 i 相邻的 超像素 编号
            # print(num[0])
            # b = np.zeros([1 + len(num[0]), img.shape[2]])
            bb = img[seg100 == i]   # 超像素编号 i 的位置的 img 的光谱向量均值
            # print(bb.shape)
            bb = np.mean(bb, axis=0)
            # print(bb.shape)

            for j in range(len(num[0])):

                img_one = img[seg100 == num[0][j]]  # # 超像素编号 num[0][j] 的位置的 img 的光谱向量均值， 就是 i 的邻域
                img_one = np.mean(img_one, axis=0)
                # c = euclidean_distances(bb, img_one)
                c = bb - img_one   # 求光谱相似性，高斯核函数
                c = np.exp(-(c * c) / 0.005)
                cc = c.sum() / len(bb)

                # cc = np.sum(c) / c.shape[1] / c.shape[0]
                # print(cc)

                # 大于 阈值 ，判定为 同一类， 然后将 num[0][j] 编号 变成跟 i ，即合并超像素块
                if cc >= threshold:

                    seg100[seg100 == num[0][j]] = i
                    index = np.where(pixel_num == num[0][j])
                    pixel_num = np.delete(pixel_num, index)
                    A[:, num[0][j]] = 0


    return seg100, A
    # name = 'A_out.mat'
    # io.savemat(name, {'a': A})
    # name = 'Segments100_out.mat'
    # io.savemat(name, {'Segments': seg100})


def smooth_pixel_two(img, seg100,  threshold):

    pixel_num = np.unique(seg100)

    def get_A(superpixel_count, segments):
        '''
         根据 segments 判定邻接矩阵
        :return:
        '''
        B = np.zeros([superpixel_count, superpixel_count], dtype=np.float32)
        # print(B.shape)
        (h, w) = segments.shape
        for m in range(h - 2):
            for n in range(w - 2):
                sub = segments[m:m + 2, n:n + 2]
                sub_max = np.max(sub).astype(np.int32)
                sub_min = np.min(sub).astype(np.int32)
                # if len(sub_set)>1:
                x1, y1 = np.where(sub == sub_max)
                x2, y2 = np.where(sub == sub_min)
                if sub_max != sub_min:
                    if (x2[0] - x1[0]) * (y2[0] - y1[0]) == 0:
                        idx1 = sub_max
                        idx2 = sub_min
                        # print(idx1, idx2)
                        if B[idx1, idx2] != 0:
                            continue
                        B[idx1, idx2] = B[idx2, idx1] = 1
        return B

    A = get_A(np.max(pixel_num) + 1, seg100)

    for i in range(np.max(pixel_num) + 1):
        # i = seg100[152,84]
        # i = seg100[432, 79]
        # print(i)
        if i in pixel_num:
            # print(i)
            num = np.where(A[i, :] == 1)   # 和 i 相邻的 超像素 编号
            # print(num[0])

            # b = np.zeros([1 + len(num[0]), img.shape[2]])
            bb = img[seg100 == i]   # 超像素编号 i 的位置的 img 的光谱向量均值
            # print(bb.shape)
            bb = np.mean(bb, axis=0)
            # print(bb.shape)

            for j in range(len(num[0])):

                img_one = img[seg100 == num[0][j]]  # # 超像素编号 num[0][j] 的位置的 img 的光谱向量均值， 就是 i 的邻域
                img_one = np.mean(img_one, axis=0)
                # c = euclidean_distances(bb, img_one)
                c = bb - img_one   # 求光谱相似性，高斯核函数
                # c = np.exp(-(c * c) / 0.005)  # for paviau
                # c = np.exp(-(c * c) / 0.005)  # for IndianPines
                # c = np.exp(-(c * c) / 0.005)  # for sa
                c = np.exp(-(c * c) / 0.005)  # for  houston2013
                # c = np.exp(-(c * c) / 0.005)  # for  whu_hanchuan
                cc = c.sum() / len(bb)

                # cc = np.sum(c) / c.shape[1] / c.shape[0]
                # print(cc, threshold)


                # 大于 阈值 ，判定为 同一类， 然后将 num[0][j] 编号 变成跟 i ，即合并超像素块
                if cc >= threshold:

                    seg100[seg100 == num[0][j]] = i
                    index = np.where(pixel_num == num[0][j])
                    pixel_num = np.delete(pixel_num, index)
                    A[:, num[0][j]] = 0
            # quit()

    return seg100, A


def smooth_pixel_window(img, seg100,  threshold):

    pixel_num = np.unique(seg100)

    def get_A(superpixel_count, segments):
        '''
         根据 segments 判定邻接矩阵
        :return:
        '''
        B = np.zeros([superpixel_count, superpixel_count], dtype=np.float32)
        # print(B.shape)
        (h, w) = segments.shape
        for m in range(h - 2):
            for n in range(w - 2):
                sub = segments[m:m + 2, n:n + 2]
                sub_max = np.max(sub).astype(np.int32)
                sub_min = np.min(sub).astype(np.int32)
                # if len(sub_set)>1:
                if sub_max != sub_min:
                    idx1 = sub_max
                    idx2 = sub_min
                    # print(idx1, idx2)
                    if B[idx1, idx2] != 0:
                        continue
                    B[idx1, idx2] = B[idx2, idx1] = 1
        return B

    A = get_A(np.max(pixel_num) + 1, seg100)

    window = 25
    half_window = window // 2
    h, w = seg100.shape
    print(h // 25)


    for m in range(0, h, window):
        for n in range(0, w, window):

            print(m, n)
            mm = m + window
            nn = n+window
            if mm > h:
                mm = h
            if nn > w:
                nn = w
            seg_one = seg100[m:mm, n:nn]

            pixel_num = np.unique(seg_one)
            print(pixel_num,len(pixel_num))
            b = np.zeros([len(pixel_num), img.shape[2]])
            print(b.shape)

            kk = 0
            for k in pixel_num:

                bb = img[seg100 == k]  # 超像素编号 i 的位置的 img 的光谱向量均值
                bb = np.mean(bb, axis=0)
                b[kk, :] = bb
                kk = kk + 1

            c = euclidean_distances(b, b)
            print(c)

            x, y = np.where(c < 0.3)
            print(x, y)

            quit()


    quit()



    for i in range(np.max(pixel_num) + 1):
        # i = seg100[324,122]
        if i in pixel_num:
            # print(i)
            num = np.where(A[i, :] == 1)   # 和 i 相邻的 超像素 编号
            # print(num[0])

            # b = np.zeros([1 + len(num[0]), img.shape[2]])
            bb = img[seg100 == i]   # 超像素编号 i 的位置的 img 的光谱向量均值
            # print(bb.shape)
            bb = np.mean(bb, axis=0)
            # print(bb.shape)

            for j in range(len(num[0])):

                img_one = img[seg100 == num[0][j]]  # # 超像素编号 num[0][j] 的位置的 img 的光谱向量均值， 就是 i 的邻域
                img_one = np.mean(img_one, axis=0)
                # c = euclidean_distances(bb, img_one)
                c = bb - img_one   # 求光谱相似性，高斯核函数
                c = np.exp(-(c * c) / 0.005)
                cc = c.sum() / len(bb)

                # cc = np.sum(c) / c.shape[1] / c.shape[0]
                # print(cc)

                # 大于 阈值 ，判定为 同一类， 然后将 num[0][j] 编号 变成跟 i ，即合并超像素块
                if cc >= threshold:

                    seg100[seg100 == num[0][j]] = i
                    index = np.where(pixel_num == num[0][j])
                    pixel_num = np.delete(pixel_num, index)
                    A[:, num[0][j]] = 0

    return seg100, A

if __name__ == '__main__':

    Datasets = "PaviaU"
    FOLDER = './Datasets/'

    img, img_original, gt, LABEL_VALUES, IGNORED_LABELS, RGB_BANDS, palette = get_dataset(Datasets, FOLDER)
    img_pca = pca_img(img, img.shape[0], img.shape[1], img.shape[2], 3)

    name111 = 'Segments_30.mat'
    data111 = io.loadmat(name111)
    seg = data111['Segments']
    print(seg.shape)
    # seg200 = np.expand_dims(seg200,2)
    # print(seg200.shape)
    # name111 = 'Segments100.mat'
    # data111 = io.loadmat(name111)
    # seg100 = data111['Segments']
    # print(seg100.shape)
    # seg100 = np.expand_dims(seg100,2)
    # print(seg100.shape)

    # seg = np.dstack((seg100,seg200))
    # print(seg.shape)

    # segone = seg100[306:355,109:138]
    # prediction_data2 = pd.DataFrame(segone)
    # data2 = pd.ExcelWriter('Seg100.xlsx')  # 写入Excel文件
    # prediction_data2.to_excel(data2, 'Seg', float_format='%.5f')  # ‘page_1’是写入excel的sheet名
    # data2.save()
    #
    # segone = seg200[306:355,109:138]
    # prediction_data2 = pd.DataFrame(segone)
    # data2 = pd.ExcelWriter('Seg200.xlsx')  # 写入Excel文件
    # prediction_data2.to_excel(data2, 'Seg', float_format='%.5f')  # ‘page_1’是写入excel的sheet名
    # data2.save()
    # name111 = 'A.mat'
    # data111 = io.loadmat(name111)
    # A = data111['a']
    # print(A.shape)

    smooth_pixel_window(img, seg, 0.4)



