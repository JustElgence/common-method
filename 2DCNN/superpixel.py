import cv2
import numpy as np

def slic1(img,out_name):
    ### SLIC 算法
    # 初始化slic项，超像素平均尺寸20（默认为10），平滑因子20
    slic = cv2.ximgproc.createSuperpixelSLIC(img, 101, region_size=10, ruler=20.0)
    slic.iterate(10)  # 迭代次数，越大效果越好
    mask_slic = slic.getLabelContourMask()  # 获取Mask，超像素边缘Mask==1
    label_slic = slic.getLabels()  # 获取超像素标签
    number_slic = slic.getNumberOfSuperpixels()  # 获取超像素数目
    mask_inv_slic = cv2.bitwise_not(mask_slic)
    img_slic = cv2.bitwise_and(img, img, mask=mask_inv_slic)  # 在原图上绘制超像素边界

    color_img = np.zeros((img.shape[0], img.shape[1], 3), np.uint8)
    color_img[:] = (0, 255, 0)
    result_ = cv2.bitwise_and(color_img, color_img, mask=mask_slic)
    print(img_slic.shape,result_.shape)
    result = cv2.add(img_slic, result_, dtype = cv2.CV_32F)

    cv2.imwrite('./result/superpixel_out/cat_SLIC_' + out_name + '.png', result)

def seed(img,out_name):

    ### SEEDS 算法
    # 初始化seeds项，注意图片长宽的顺序
    seeds = cv2.ximgproc.createSuperpixelSEEDS(img.shape[1], img.shape[0], img.shape[2], 2000, 15, 3, 5, True)  # 第4个参数是  超像素的个数
    seeds.iterate(img, 10)  # 输入图像大小必须与初始化形状相同，迭代次数为10
    mask_seeds = seeds.getLabelContourMask()
    label_seeds = seeds.getLabels()
    number_seeds = seeds.getNumberOfSuperpixels()
    mask_inv_seeds = cv2.bitwise_not(mask_seeds)
    img_seeds = cv2.bitwise_and(img, img, mask=mask_inv_seeds)

    color_img = np.zeros((img.shape[0], img.shape[1], 3), np.uint8)
    color_img[:] = (0, 255, 0)
    result_ = cv2.bitwise_and(color_img, color_img, mask=mask_seeds)
    result = cv2.add(img_seeds, result_)

    cv2.imwrite('../data/superpixel_out/cat_SEEDS_' + out_name + '.png', result)

def lsc(img,out_name):

    lsc = cv2.ximgproc.createSuperpixelLSC(img, 20)
    lsc.iterate(10)
    mask_lsc = lsc.getLabelContourMask()
    label_lsc = lsc.getLabels()
    number_lsc = lsc.getNumberOfSuperpixels()
    mask_inv_lsc = cv2.bitwise_not(mask_lsc)
    img_lsc = cv2.bitwise_and(img, img, mask=mask_inv_lsc)

    color_img = np.zeros((img.shape[0], img.shape[1], 3), np.uint8)
    color_img[:] = (0, 255, 0)
    result_ = cv2.bitwise_and(color_img, color_img, mask=mask_lsc)
    result = cv2.add(img_lsc, result_)

    cv2.imwrite('../data/superpixel_out/cat_LSC_' + out_name + '.png', result)


if __name__ == '__main__':

    slic_region_size = 30
    number = 1
    imgl_name = "l" + str(number)
    imgr_name = 'r' + str(number)
    iml = cv2.imread('../data/data/' + imgl_name + '.jpg')  # 左图
    imr = cv2.imread('../data/data/' + imgr_name + '.jpg')  # 右图

    # 左图
    slic(iml,imgl_name,slic_region_size)
    seed(iml,imgl_name)
    lsc(iml,imgl_name)
    # 右图
    slic(imr,imgr_name,slic_region_size)
    seed(imr,imgr_name)
    lsc(imr,imgr_name)
    print('运行结束')

