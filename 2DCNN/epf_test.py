

import cv2 as cv
import numpy as np
import torch


a = torch.tensor([
             [[[0,0,0],
               [0,0,0],
               [0,1,0]],
              [[0,1,2],
               [0,0,0],
               [0,1,0]],
              [[0, 1, 2],
               [0, 0, 0],
               [0, 0, 0]]],

             [[[0, 0, 0],
              [0, 0, 0],
              [0, 1, 0]],
             [[0, 1, 2],
              [0, 0, 0],
              [0, 1, 0]],
             [[0, 1, 2],
              [0, 0, 0],
              [0, 0, 0]]],

    [[[0, 0, 0],
      [0, 0, 0],
      [0, 1, 0]],
     [[0, 1, 2],
      [0, 0, 0],
      [0, 1, 0]],
     [[0, 1, 2],
      [0, 0, 0],
      [0, 0, 0]]],

    [[[0, 0, 0],
      [0, 0, 0],
      [0, 1, 0]],
     [[0, 1, 2],
      [0, 0, 0],
      [0, 1, 0]],
     [[0, 1, 2],
      [0, 0, 0],
      [0, 0, 0]]],

    [[[0, 0, 0],
      [0, 0, 0],
      [0, 1, 0]],
     [[0, 1, 2],
      [0, 0, 0],
      [0, 1, 0]],
     [[0, 1, 2],
      [0, 0, 0],
      [0, 0, 0]]],
               ])
b = torch.tensor([
    [[[0, 0, 0, 0, 0, 0, 0, 1, 0],
      [0, 1, 2, 0, 0, 0, 0, 1, 0],
      [0, 1, 2, 0, 0, 0, 0, 0, 0]],

    [[0, 0, 0, 0, 0, 0, 0, 1, 0],
     [0, 1, 2, 0, 0, 0, 0, 1, 0],
     [0, 1, 2, 0, 0, 0, 0, 0, 0]],

    [[0, 0, 0, 0, 0, 0, 0, 1, 0],
     [0, 1, 2, 0, 0, 0, 0, 1, 0],
     [0, 1, 2, 0, 0, 0, 0, 0, 0]]],

    [[[0, 0, 0, 0, 0, 0, 0, 1, 0],
     [0, 1, 2, 0, 0, 0, 0, 1, 0],
     [0, 1, 2, 0, 0, 0, 0, 0, 0]],

    [[0, 0, 0, 0, 0, 0, 0, 1, 0],
     [0, 1, 2, 0, 0, 0, 0, 1, 0],
     [0, 1, 2, 0, 0, 0, 0, 0, 0]],
    [[0, 0, 0, 0, 0, 0, 0, 1, 0],
     [0, 1, 2, 0, 0, 0, 0, 1, 0],
     [0, 1, 2, 0, 0, 0, 0, 0, 0]]]])
# print(b,b.size())
# c = b.permute(0, 3, 1, 2)
print(b,b.size())
b[1,1,1] = 2
aa = [[0,0],[0,2],[1,2]]

bb = [[1,1],[1,1],[1,1]]
b[aa] = 1
print(b)
b[aa] = b[bb]
print(b)
quit()
# index = torch.LongTensor([])

# c = torch.gather(b,2)
print(c.size())

quit()






















src=cv.imread(r'D:\1111.jpg')
cv.imshow('xiaojiejie',src)
print(src)
print("数据类型", type(src))  # 打印数组数据类型
print("数组元素数据类型：", src.dtype)  # 打印数组元素数据类型
print("数组元素总数：", src.size)  # 打印数组尺寸，即数组元素总数
print("数组形状：", src.shape)  # 打印数组形状
print("数组的维度数目", src.ndim)  # 打印数组的维度数目

dst = cv.bilateralFilter(src, d=0, sigmaColor=3, sigmaSpace=15)

cv.imshow("bi_demo", dst)
cv.waitKey(0)


# Gaussian filter
def gaussian_filter(img, K_size=3, sigma=1.3):

    H, W, C = img.shape
    ## Zero padding
    pad = K_size // 2
    out = np.zeros((H + pad * 2, W + pad * 2, C), dtype=np.float)
    out[pad: pad + H, pad: pad + W] = img.copy().astype(np.float)

    ## prepare Kernel

    K = np.zeros((K_size, K_size), dtype=np.float)

    for x in range(-pad, -pad + K_size):

        for y in range(-pad, -pad + K_size):

            K[y + pad, x + pad] = np.exp( -(x ** 2 + y ** 2) / (2 * (sigma ** 2)))

    K /= (2 * np.pi * sigma * sigma)

    K /= K.sum()

    tmp = out.copy()

    # filtering

    for y in range(H):

        for x in range(W):

            for c in range(C):

                out[pad + y, pad + x, c] = np.sum(K * tmp[y: y + K_size, x: x + K_size, c])

    out = np.clip(out, 0, 255)

    out = out[pad: pad + H, pad: pad + W].astype(np.uint8)

    return out

# Read image

img = cv.imread("../paojie.jpg")

# Gaussian Filter

out = gaussian_filter(img, K_size=3, sigma=1.3)

# Save result

cv.imwrite("out.jpg", out)

cv.imshow("result", out)

cv.waitKey(0)

