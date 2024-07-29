import numpy as np
import cv2
import math
import pandas as pd

def getClosenessWeight(sigma_g, H, W):
    # 计算空间距离权重模板
    r, c = np.mgrid[0:H:1, 0:W:1]  # 构造三维表
    r -= int((H - 1) / 2)
    c -= int((W - 1) / 2)
    closeWeight = np.exp(-(np.power(r, 2) + np.power(c, 2)) / sigma_g * sigma_g)
    return closeWeight


def jointBLF(I, H, W, sigma_g, sigma_d, borderType=cv2.BORDER_DEFAULT):
    # 构建空间距离权重模板
    closenessWeight = getClosenessWeight(sigma_g, H, W)
    print(closenessWeight)
    # quit()

    # 对I进行高斯平滑
    # Ig = cv2.GaussianBlur(I, (W, H), sigma_g)
    Ig = I

    # 模板的中心点位置
    cH = int((H - 1) / 2)
    cW = int((W - 1) / 2)

    # 对原图和高斯平滑的结果扩充边界
    # Ip = cv2.copyMakeBorder(I, cH, cH, cW, cW, borderType)
    # Igp = cv2.copyMakeBorder(Ig, cH, cH, cW, cW, borderType)
    # print(Ip.shape)

    # 图像矩阵的行数和列数
    rows, cols = I.shape
    i, j = 0, 0

    # 联合双边滤波的结果
    jblf = np.zeros(I.shape, np.float64)
    for r in range(cH,  rows - cH, 1):
        for c in range(cW,  cols - cW, 1):
            # 当前位置的值
            pixel = Ig[r][c]

            # 当前位置的邻域
            rTop, rBottom = r - cH, r + cH
            cLeft, cRight = c - cW, c + cW

            # 从 Igp 中截取该邻域，用于构建相似性权重模板
            region = Ig[rTop: rBottom + 1, cLeft: cRight + 1]

            # 通过上述邻域，构建该位置的相似性权重模板
            similarityWeight = np.exp(-np.power(region - pixel, 2.0) / sigma_d * sigma_d)

            # 相似性权重模板和空间距离权重模板相乘
            weight = closenessWeight * similarityWeight

            # 将权重归一化
            weight = weight / np.sum(weight)

            # 权重模板和邻域对应位置相乘并求和
            jblf[i][j] = np.sum(I[rTop:rBottom + 1, cLeft:cRight + 1] * weight)

            j += 1
        j = 0
        i += 1
    return jblf


if __name__ == '__main__':

    I = cv2.imread('aaaa.PNG.jpg', cv2.IMREAD_GRAYSCALE)
    # 将8位图转换为浮点型
    # fI = I.astype(np.float64)

    # 联合双边滤波，返回值的数据类型为浮点型
    jblf = jointBLF(I, 5, 5, 7, 2)
    jblf = np.round(jblf)
    jblf = jblf.astype(np.uint8)

    cv2.imshow('origin', I)
    cv2.imshow('jblf', jblf)
    cv2.waitKey(0)
    cv2.destroyAllWindows()