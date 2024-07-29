# -*- coding: utf-8 -*-
import torch
torch.set_printoptions(profile="full")
import torch.nn as nn
from collections import Counter
import itertools
import numpy as np
np.set_printoptions(threshold=np.inf)  # 加上这一句


class DeformConv2D_EPF(nn.Module):

    def __init__(self, inc, outc, kernel_size=3, padding=1, stride=1):
        super(DeformConv2D_EPF, self).__init__()

        self.conv = nn.Conv2d(inc, outc, kernel_size=3, stride=1, padding=1)

    def forward(self, x):

        y = x

        def bilateral_filter22(image):

            img = image.detach()  # 截断反向传播
            batch_size = img.shape[0]
            n_band = img.shape[1]
            patch_size = img.shape[2]
            img = img.permute(0, 2, 3, 1)
            img_hyper = img[:,:,:,:n_band - 2]
            img_superpixel = img[:,:,:,n_band - 2]
            train_or_test = img[0,:,:,n_band - 1]

            img_superpixel = img_superpixel.cpu().numpy()

            for i in range(img.shape[0]):

                img_superpixel_one = img_superpixel[i, :, :]
                centrel_pixel = img_superpixel_one[patch_size // 2, patch_size // 2]
                # # 随机添加 centrel_pixel   会增加 不需要 偏移的 个数，就是0 减少
                # print('train,test not print')
                img_superpixel_one1 = img_superpixel_one.flatten()
                #
                # indices = np.random.choice(np.arange(img_superpixel_one1.size), replace=False,
                #                            size=int(img_superpixel_one1.size * 0.05))   # for paviau  0.02 for 21  不要改
                                                                                        # for paviau  0.03 for 19  17  不要改
                                                                                        # for paviau  0.05 for 15  不要改
                indices = np.random.choice(np.arange(img_superpixel_one1.size), replace=False,
                                           size=int(img_superpixel_one1.size * 0.08))   # for hou  0.08 for 13115
                # indices=np.random.choice(np.arange(img_superpixel_one1.size), replace=False,
                #                              size=int(img_superpixel_one1.size * 0.1))  # for indi
                # indices=np.random.choice(np.arange(img_superpixel_one1.size), replace=False,
                #                              size=int(img_superpixel_one1.size * 0.2))  # for sa   选择0.2


                img_superpixel_one1[indices] = centrel_pixel
                img_superpixel_one = img_superpixel_one1.reshape(patch_size, patch_size)

                img_superpixel_one[img_superpixel_one != centrel_pixel] = 0
                img_superpixel_one[img_superpixel_one == centrel_pixel] = 1

                img_superpixel[i, :, :] = img_superpixel_one

            image_central = img_superpixel

            # 得到类别不同的像素坐标
            # coo_not_one_b is batch_size
            # coo_not_one_x 是行坐标
            # coo_not_one_y 是列坐标
            coo_not_one_b, coo_not_one_x, coo_not_one_y = np.where(image_central == 0.)
            # 得到类别相同的像素坐标
            # coo_one_b is batch_size
            # coo_one_x 是行坐标
            # coo_one_y 是列坐标
            coo_one_b, coo_one_x, coo_one_y = np.where(image_central == 1.)
            # 如果存在 类别不同的像素，就需要 生成对应的像素的位置偏移量
            if len(coo_not_one_b) > 0:
                # num_one_sum = torch.cumsum(coo_not_one_b,0)
                # coo = [(coo_not_one_b[a], coo_not_one_x[a], coo_not_one_y[a]) for a in range(coo_not_one_b.shape[0])]
                # coo1 = [(coo_one_b[a], coo_one_x[a], coo_one_y[a]) for a in range(coo_one_b.shape[0])]
                # print(coo[10:15])
                # print(len(coo), coo[:2], coo[0][0], coo[0][0].item())
                # quit()
                num1 = Counter(coo_not_one_b)  # 计算 不同类别像素的个数
                # print(num1)
                # Counter({19: 154, 5: 135, 32: 74, 51: 72, 30: 67, 43: 50, 60: 49, 7: 47, 62: 46, 44: 35, 25: 29,
                # 37: 27, 49: 25, 47: 24, 12: 20, 33:18, 20: 16, 24: 16, 22: 14, 1: 13, 6: 11, 3: 9, 27: 5, 45: 5,
                # 29: 3, 38: 3, 48: 3, 4: 2, 40: 2, 56: 2, 42: 1, 59: 1, 63: 1})
                num_not_one = np.array([num1[g] for g in range(batch_size)])  # 将 num1 中数量按照 batch_size 由 0 开始排序
                # print('num_not_one','\n', num_not_one,num_not_one.sum())
                # [0  13   0   9   2 135  11  47   0   0   0   0  20   0   0   0   0   0
                #  0 154  16   0  14   0  16  29   0   5   0   3  67   0  74  18   0   0
                #  0  27   3   0   2   0   1  50  35   5   0  24   3  25   0  72   0   0
                #  0   0   2   0   0   1  49   0  46   1] 979
                num2 = Counter(coo_one_b)   # 计算 相同类别像素的个数
                num_one = np.array([num2[g] for g in range(batch_size)])  # 将 num2 中数量按照 batch_size 由 0 开始排序
                # print('num_one','\n', num_one,num_one.sum())

                # 计算每个 batch_size 中需要偏移的像素个数,
                # 如果需要偏移的个数（不同类别的个数）小于 可以偏移的个数（同类别像素的个数），就取值 需要偏移的个数（不同类别的个数）
                num_chazhi = np.where((num_not_one - num_one) > 0, num_one, num_not_one)
                '''
                for example 
                 num_not_one:[  0  17  90 197   2  63   0   0   3   0   0  86  25   1   0  25  25   0
                             0   0   0  47 246   0   0  68  27   0  13   3   4   3   0   0   0   0
                             13   4   1  81  37   0   0  31   0   0  12   9   9 263   2   0   0   1
                          0  28   0   0   0  51  16   0   0  75] 1578
                 num_one:
                  [354 259  75  45 224 207 435 125 325 412 441 177 157 172  57 199 280 379
                   441 420 430 161  77 319  79  62 203 426 180 322  91 195 436 362 432 242
                   259 288 283 168 301 322 242 175 389 441 217 285 168  51 220 364 390 412
                   194 138 300 431 388 299 193 177  17 163] 16476
               num_chazhi:
                     [ 0 17 75 45  2 63  0  0  3  0  0 86 25  1  0 25 25  0  0  0  0 47 77  0
                       0 62 27  0 13  3  4  3  0  0  0  0 13  4  1 81 37  0  0 31  0  0 12  9
                       9 51  2  0  0  1  0 28  0  0  0 51 16  0  0 75] 1024
                '''
                # print('\n','num_chazhi','\n',num_chazhi, num_chazhi.sum())
                # quit()

                # coo_not_one_b = coo_not_one_b.reshape(len(coo_not_one_b), 1)
                # coo_not_one_x = coo_not_one_x.reshape(len(coo_not_one_x), 1)
                # coo_not_one_y = coo_not_one_y.reshape(len(coo_not_one_y), 1)
                coo_one_b = coo_one_b.reshape(len(coo_one_b), 1)  # 转成列向量
                coo_one_x = coo_one_x.reshape(len(coo_one_x), 1)
                coo_one_y = coo_one_y.reshape(len(coo_one_y), 1)
                # coo333 = np.concatenate((coo_not_one_b, coo_not_one_x, coo_not_one_y), 1)
                coo444 = np.concatenate((coo_one_b, coo_one_x, coo_one_y), 1)  # 拼接成 一个 3列的矩阵，
                # print(coo333.shape, coo444.shape)
                # quit()

                # num_one_sum = [num_one[:a].sum() for a in range(batch_size)]
                # print('num_one_sum','\n',num_one_sum,len(num_one_sum))
                # num_not_one_sum = [num_not_one[:a].sum() for a in range(batch_size)]
                # print('num_not_one_sum','\n',num_not_one_sum, len(num_not_one_sum))
                # quit()
                # 将  可以偏移的个数（同类别像素的个数）的点 坐标组合一起
                # coo2 就是 类别不同的像素 偏移到 的位置
                coo2 = [coo444[num_one[:a].sum():(num_one[:a].sum() + num_chazhi[a]), :] for a in range(batch_size)]
                coo2 = list(itertools.chain(*coo2))
                coo2 = np.array(coo2)
                # print(coo2)
                # print(coo2.shape)
                # quit()

                num_chazhi_zeros = np.where((num_not_one - num_one) > 0)[0]
                # print('num_chazhi_zeros','\n',num_chazhi_zeros, len(num_chazhi_zeros))
                # 如果需要偏移的个数（不同类别的个数）小于 可以偏移的个数（同类别像素的个数），我们做一下处理：
                # 现在是 可以偏移的个数（同类别像素的个数）太少了，所以我们重复使用 可以偏移的像素点（同类别像素），然后把重复后的这些像素点插入到 coo2 中
                if len(num_chazhi_zeros) > 0:
                    extend_num = 0
                    for h in range(len(num_chazhi_zeros)):
                        # print(h)
                        # print(num_chazhi_zeros[h])
                        a = num_chazhi_zeros[h]
                        # print(num_one[:a].sum())
                        # print((num_one[:a].sum() + num_chazhi[a]))
                        # quit()
                        coo444_one = coo444[num_one[:a].sum():num_one[:(a + 1)].sum(), :]
                        # print(coo444_one.shape)
                        # print(num_not_one[a],)
                        # print(num_one[a], int(num_not_one[a] // num_one[a]))
                        # quit()
                        coo444_one = np.tile(coo444_one, (int(num_not_one[a] // num_one[a]), 1))
                        # print(coo444_one.shape)
                        coo444_one = coo444_one[:(num_not_one[a] - num_one[a]),:]
                        # print('本次插入数据长度',coo444_one.shape[0])
                        # print('extend_num',extend_num)
                        # print('插入位置1',num_chazhi[:(a + 1)].sum() + extend_num)
                        # print('插入位置2', num_not_one[:(a + 1)].sum() + extend_num)
                        # print('插入前coo2 的长度 ', coo2.shape)
                        # coo2_b = coo2[:, 0]
                        # qqqqqq = Counter(coo2_b)
                        # num_not_one111 = np.array([qqqqqq[g] for g in range(batch_size)])
                        # print(num_not_one111, num_not_one111.sum())
                        coo2 = np.insert(coo2, (num_chazhi[:(a + 1)].sum() + extend_num), coo444_one, 0)

                        # coo2_b = coo2[:, 0]
                        # qqqqqq = Counter(coo2_b)
                        # num_not_one111 = np.array([qqqqqq[g] for g in range(batch_size)])
                        # print(num_not_one111,num_not_one111.sum())

                        extend_num = extend_num + coo444_one.shape[0]
                        # print('插入后coo2 的长度 ',coo2.shape)
                        # print('extend_num',extend_num)
                        # print(coo2.shape)
                        # quit()

                        # quit()
                # coo2_b = coo2[:, 0]
                # qqqqqq = Counter(coo2_b)
                # num_not_one = np.array([qqqqqq[g] for g in range(batch_size)])
                # print(num_not_one,num_not_one.sum())
                # quit()
                #
                # coo3 = [coo333[num_not_one[:a].sum():(num_not_one[:a].sum() + num_chazhi[a]), :] for a in
                #         range(batch_size)]
                # coo3 = list(itertools.chain(*coo3))
                # coo3 = np.array(coo3)
                # # print(coo3.shape)
                # quit()

                # coo3_b = coo333[:, 0]
                # coo3_x = coo333[:, 1]
                # coo3_y = coo333[:, 2]
                # coo3_b = coo_not_one_b
                # coo3_x = coo_not_one_x
                # coo3_y = coo_not_one_y
                coo2_b = coo2[:, 0]
                coo2_x = coo2[:, 1]
                coo2_y = coo2[:, 2]
                image = image.permute(0, 2, 3, 1)
                # print(image.shape)
                # start = torch.cuda.Event(enable_timing=True)
                # end = torch.cuda.Event(enable_timing=True)
                # start.record()

                # for j in range(coo2.shape[0]):
                #
                #     image[coo3[j][0].item(), :, coo3[j][1].item(), coo3[j][2].item()] = image[coo2[j][0].item(), :, coo2[j][1].item(), coo2[j][2].item()]
                #
                # end.record()
                # # Waits for everything to finish running
                # torch.cuda.synchronize()
                # print(start.elapsed_time(end))
                # 将 组合成的 coo2 中的像素点 替换到 coo_not_one 中
                img_hyper[coo_not_one_b, coo_not_one_x, coo_not_one_y] = img_hyper[coo2_b, coo2_x, coo2_y]
                # img_hyper = img_hyper.permute(0, 3, 1, 2)
                return img_hyper
            else:
                return img_hyper

        x_offset = bilateral_filter22(y).cuda()
        x_offset = x_offset.permute(0, 3, 1, 2)

        out = self.conv(x_offset)  # 正常卷积

        return out
