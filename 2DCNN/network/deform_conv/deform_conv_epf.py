# -*- coding: utf-8 -*-
import torch
torch.set_printoptions(profile="full")
import torch.nn as nn
import cv2 as cv
from collections import Counter
import math
import itertools
import numpy as np
np.set_printoptions(threshold=np.inf)  # 加上这一句
import pandas as pd
from scipy.spatial import distance
from utils import bilateralFilter_img
from utils import gaus_kernel


class DeformConv2D_EPF(nn.Module):

    def __init__(self, inc, outc, kernel_size=3, padding=1, stride=1, bias=None,
                 lr_ratio=1.0):  # input is : 64, 128, 3, 1, 1
        super(DeformConv2D_EPF, self).__init__()
        self.kernel_size = kernel_size
        # print(kernel_size)
        # quit()
        self.padding = padding
        self.stride = stride
        self.zero_padding = nn.ZeroPad2d(padding)  # ZeroPad2d : 使用0填充输入tensor的边界

        self.offset_conv = nn.Conv2d(1, 2*kernel_size * kernel_size, kernel_size=3, padding=1, stride=stride)
        # self.offset_conv1 = nn.Conv2d(kernel_size * kernel_size, 2 * kernel_size * kernel_size, kernel_size=3, padding=1, stride=stride)
        nn.init.constant_(self.offset_conv.weight, 0)  # the offset learning are initialized with zero weights
        # nn.init.constant_(self.offset_conv1.weight, 1)  # the offset learning are initialized with zero weights
        self.offset_conv.register_backward_hook(self._set_lr)
        # self.offset_conv1.register_backward_hook(self._set_lr)

        self.conv = nn.Conv2d(inc, outc, kernel_size=kernel_size, stride=kernel_size, bias=bias)

        self.lr_ratio = lr_ratio

    def _set_lr(self, module, grad_input, grad_output):
        # print('grad input:', grad_input)
        new_grad_input = []

        for i in range(len(grad_input)):
            if grad_input[i] is not None:
                new_grad_input.append(grad_input[i] * self.lr_ratio)
            else:
                new_grad_input.append(grad_input[i])

        new_grad_input = tuple(new_grad_input)
        # print('new grad input:', new_grad_input)
        return new_grad_input

    def forward(self, x):
        y = x
        def bilateralFilter(data, threshold):

            img = data.cpu().detach().numpy()
            batch_size = img.shape[0]
            n_band = img.shape[1]
            patch_size = img.shape[2]
            central_pixel = patch_size // 2
            # print(batch_size,n_band,patch_size)
            # print(img[0,:,0,0])
            imgg = np.zeros([batch_size,patch_size,patch_size,1])

            for i in range(batch_size):
                img1 = img[i, :, :, :]
                img1 = img1.transpose((1, 2, 0))
                # print(img1.shape)
                inputmap_bF = bilateralFilter_img(img1, 3, 20, 20)

                inputmap_bF_central_pixel_abs = abs(inputmap_bF - inputmap_bF[central_pixel, central_pixel])
                # print(inputmap_bF_central_pixel_abs)
                inputmap_bF_central_pixel_abs = np.exp(-(inputmap_bF_central_pixel_abs ** 2) / (threshold))
                inputmap_bF_central_pixel_abs = np.reshape(inputmap_bF_central_pixel_abs,(inputmap_bF_central_pixel_abs.shape[0], inputmap_bF_central_pixel_abs.shape[1], 1))
                imgg[i, :, :, :] = inputmap_bF_central_pixel_abs

            imgg = imgg.transpose((0,3,1, 2))
            imgg = torch.from_numpy(imgg)

            return imgg

        xx = bilateralFilter(y, 100).type(torch.FloatTensor).cuda()

        offset = self.offset_conv(xx)
        dtype = offset.data.type()
        # print('offset.data.type() : ', dtype)
        ks = self.kernel_size
        # print('ks = self.kernel_size,ks is : ', ks)
        N = offset.size(1) // 2
        # print('run deform_conv, N = offset.size(1) // 2, the N size : ', N)
        # quit()

        # Change offset's order from [x1, x2, ..., y1, y2, ...] to [x1, y1, x2, y2, ...]
        # Codes below are written to make sure same results of MXNet implementation.
        # You can remove them, and it won't influence the module's performance.
        # offsets_index = torch.cat([torch.arange(0, 2 * N, 2), torch.arange(1, 2 * N + 1, 2)]).type_as(x).long()
        # offsets_index.requires_grad = False
        # offsets_index = offsets_index.unsqueeze(dim=0).unsqueeze(dim=-1).unsqueeze(dim=-1).expand(*offset.size())
        # offset = torch.gather(offset, dim=1, index=offsets_index)
        # ------------------------------------------------------------------------

        if self.padding:
            # print('对输入的 X 进行 0 填充')
            x = self.zero_padding(x)
            # print('run deform_conv, 对输入的 X 进行 0 填充后，the input data X  size', x.size())
            '''
            如果输入 是一个 3*3  则补 0  成 5*5 的
            这样的话，可以正常卷积
            '''
            # print('***************************分隔符**************************')

        # (b, 2N, h, w)
        p = self._get_p(offset, dtype)
        # print('self._get_p, p size is : ', p.size())

        # (b, h, w, 2N)  就是 维度 转换
        p = p.contiguous().permute(0, 2, 3, 1)
        # print('p.contiguous().permute',p.size(),'\n','p is : ','\n',p)    # 《Deformable Convolutional Networks》中 公式1 或者公式2 中的 p
        # quit()
        # 2020.9.21 晚 理解到这
        """
            if q is float, using bilinear interpolate(双线性插值), it has four integer corresponding position.（四个整数相关位置）
            The four position is left top, right top, left bottom, right bottom, defined as q_lt, q_rb, q_lb, q_rt
        """
        # (b, h, w, 2N)
        '''
        简单来说detach就是截断反向传播的梯度流
        '''
        '''
        floor() 返回数字的下舍整数。
        math.floor(-45.17) :  -46.0
        math.floor(100.12) :  100.0
        math.floor(100.72) :  100.0
        '''
        q_lt = p.detach().floor()  # 不更新 参数  ，， 向下取整，就是舍弃小数
        # print('q_lt = p.detach().floor(), q_lt : ',q_lt.size(),'\n','q_lt is : ','\n',q_lt,'\n','q_lt[..., :N] is : ','\n',q_lt[..., :N])
        # quit()
        """
            Because the shape of x is N, b, h, w, the pixel position is (y, x)
            *┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄→y
            ┊  .(y, x)   .(y+1, x)
            ┊   
            ┊  .(y, x+1) .(y+1, x+1)
            ┊
            ↓
            x

            For right bottom point, it'x = left top'y + 1, it'y = left top'y + 1
        """
        q_rb = q_lt + 1
        # print('q_rb is : ',q_rb.size(),q_rb)

        """
            x.size(2) is h, x.size(3) is w, make 0 <= p_y <= h - 1, 0 <= p_x <= w-1
        """
        ''' 
        torch.clamp(input, min, max, out=None) → Tensor

        将输入input张量每个元素的夹紧到区间 [min,max][min,max]，并返回结果到一个新张量。

              | min,   if x_i < min
        y_i = | x_i,   if min <= x_i <= max
              | max,   if x_i > max

        example:
        a=torch.randint(low=0,high=10,size=(10,1))
        print(a)
        a=torch.clamp(a,3,9)
        print(a)
        '''
        '''
        torch.cat是将两个张量（tensor）拼接在一起，cat是concatnate的意思，即拼接，联系在一起。
        使用torch.cat((A,B),dim)时，除拼接维数dim数值可不同外其余维数数值需相同，方能对齐。
        '''
        '''
        [m : ] 代表列表中的第m+1项到最后一项
        [ : n] 代表列表中的第一项到第n项
        '''
        # 目标坐标需要在图片最大坐标范围内，将目标坐标进行切割限制
        q_lt = torch.cat([torch.clamp(q_lt[..., :N], 0, x.size(2) - 1), torch.clamp(q_lt[..., N:], 0, x.size(3) - 1)],
                         dim=-1).long()
        # print('q_lt = torch.cat , q_lt : ', q_lt.size(),q_lt)
        # quit()

        """
            x.size(2) is h, x.size(3) is w, make 0 <= p_y <= h - 1, 0 <= p_x <= w-1
        """
        q_rb = torch.cat([torch.clamp(q_rb[..., :N], 0, x.size(2) - 1), torch.clamp(q_rb[..., N:], 0, x.size(3) - 1)],
                         dim=-1).long()
        # print('q_rb = torch.cat , q_rb : ', q_rb.size())

        """
            For the left bottom point, it'x is equal to right bottom, it'y is equal to left top
            Therefore, it's y is from q_lt, it's x is from q_rb
        """
        q_lb = torch.cat([q_lt[..., :N], q_rb[..., N:]], -1)
        # print('q_lb = torch.cat , q_lb : ', q_lb.size())
        """
            y from q_rb, x from q_lt
            For right top point, it's x is equal t to left top, it's y is equal to right bottom 
        """
        q_rt = torch.cat([q_rb[..., :N], q_lt[..., N:]], -1)
        # print('q_rt = torch.cat , q_rt : ', q_rt.size())
        # quit()
        """
            find p_y <= padding or p_y >= h - 1 - padding, find p_x <= padding or p_x >= x - 1 - padding

            This is to find the points in the area where the pixel value is meaningful.
        """
        # (b, h, w, N)
        mask = torch.cat([p[..., :N].lt(self.padding) + p[..., :N].gt(x.size(2) - 1 - self.padding),
                          p[..., N:].lt(self.padding) + p[..., N:].gt(x.size(3) - 1 - self.padding)], dim=-1).type_as(p)
        mask = mask.detach()
        # print('mask:', mask)

        floor_p = torch.floor(p)
        # print('floor_p = ', floor_p)

        """
           when mask is 1, take floor_p;
           when mask is 0, take original p.
           When the point in the padding area, interpolation is not meaningful and we can take the nearest
           point which is the most possible to have meaningful value.
        """
        p = p * (1 - mask) + floor_p * mask
        p = torch.cat([torch.clamp(p[..., :N], 0, x.size(2) - 1), torch.clamp(p[..., N:], 0, x.size(3) - 1)], dim=-1)

        """
            In the paper, G(q, p) = g(q_x, p_x) * g(q_y, p_y)
            g(a, b) = max(0, 1-|a-b|)
        """
        # bilinear kernel (b, h, w, N)
        g_lt = (1 + (q_lt[..., :N].type_as(p) - p[..., :N])) * (1 + (q_lt[..., N:].type_as(p) - p[..., N:]))
        g_rb = (1 - (q_rb[..., :N].type_as(p) - p[..., :N])) * (1 - (q_rb[..., N:].type_as(p) - p[..., N:]))
        g_lb = (1 + (q_lb[..., :N].type_as(p) - p[..., :N])) * (1 - (q_lb[..., N:].type_as(p) - p[..., N:]))
        g_rt = (1 - (q_rt[..., :N].type_as(p) - p[..., :N])) * (1 + (q_rt[..., N:].type_as(p) - p[..., N:]))

        # print('g_lt size is ', g_lt.size(),g_lt)
        # quit()
        # print('g_rb size is ', g_rb.size())
        # print('g_lb size is ', g_lb.size())
        # print('g_rt size is ', g_rt.size())
        # # quit()
        # print('g_lt unsqueeze size:', g_lt.unsqueeze(dim=1).size())

        # (b, c, h, w, N)
        # print(' the input x.size() : ', x.size())
        x_q_lt = self._get_x_q(x, q_lt, N)
        x_q_rb = self._get_x_q(x, q_rb, N)
        x_q_lb = self._get_x_q(x, q_lb, N)
        x_q_rt = self._get_x_q(x, q_rt, N)
        # print('x_q_lt size is ', x_q_lt.size())
        # print('x_q_rb size is ', x_q_rb.size())
        # print('x_q_lb size is ', x_q_lb.size())
        # print('x_q_rt size is ', x_q_rt.size())
        """
            In the paper, x(p) = ΣG(p, q) * x(q), G is bilinear kernal
        """
        # (b, c, h, w, N)
        x_offset = g_lt.unsqueeze(dim=1) * x_q_lt + \
                   g_rb.unsqueeze(dim=1) * x_q_rb + \
                   g_lb.unsqueeze(dim=1) * x_q_lb + \
                   g_rt.unsqueeze(dim=1) * x_q_rt

        """
            x_offset is kernel_size * kernel_size(N) times x. 
        """
        x_offset = self._reshape_x_offset(x_offset, ks)
        # print('x_offset is : ', x_offset.size())
        # 2.Get all integrated pixels into  a new image as input of  next layer.The rest is same as CNN
        # quit()

        out = self.conv(x_offset)
        # print('out size is : ', out.size())
        # quit()
        return out

    ###################生成固定的 p_n 和 p_0 #############################
    def _get_p_n(self, N, dtype):
        """
            In torch 0.4.1 grid_x, grid_y = torch.meshgrid([x, y])
            In torch 1.0   grid_x, grid_y = torch.meshgrid(x, y)
        """
        '''
        x1 ,y1 = torch.meshgrid(x,y)
        参数是两个，第一个参数我们假设是x，第二个参数假设就是y
        输出的是两个tensor，size就是x.size * y.size（行数是x的个数，列数是y的个数）
        具体输出看下面
        注意：两个参数的数据类型要相同，要么都是float，要么都是int，否则会报错。    
        '''
        '''
        >>> z=torch.arange(1,6)
        >>> z
            tensor([1, 2, 3, 4, 5])
        >>> z.dtype
            torch.int64

            torch.arange(start=1, end=6)的结果并不包含end。  
        '''
        p_n_x, p_n_y = torch.meshgrid(
            [torch.arange(-(self.kernel_size - 1) // 2, (self.kernel_size - 1) // 2 + 1),
             torch.arange(-(self.kernel_size - 1) // 2, (self.kernel_size - 1) // 2 + 1)])  # kernel_size is 3
        # 就是 等价于 torch.arange(-1,2)
        # print('p_n_x is : ','\n',p_n_x,'\n',' p_n_y is : ','\n',p_n_y)
        '''
        p_n_x,p_n_y is :
                        tensor([[-1, -1, -1],
                                [ 0,  0,  0],
                                [ 1,  1,  1]])
                        tensor([[-1,  0,  1],
                                [-1,  0,  1],
                                [-1,  0,  1]])     
        '''

        # (2N, 1)
        p_n = torch.cat([torch.flatten(p_n_x), torch.flatten(p_n_y)], 0)
        # print('p_n.size() is : ',p_n.size(),'\n','p_n is : ','\n',p_n)
        '''
        p_n.size() is :  torch.Size([18])
        tensor([-1, -1, -1,  0,  0,  0,  1,  1,  1, -1,  0,  1, -1,  0,  1, -1,  0,  1])
        '''
        p_n = p_n.view(1, 2 * N, 1, 1).type(dtype)
        p_n.requires_grad = False
        # print('requires_grad:', p_n.requires_grad)
        # print('p_n.size() is : ',p_n.size(),'\n',p_n)
        '''
        p_n.size() is :  torch.Size([1, 18, 1, 1])
        '''
        # quit()
        return p_n

    def _get_p_0(self, h, w, N, dtype):
        p_0_x, p_0_y = torch.meshgrid([
            torch.arange(1, h * self.stride + 1, self.stride),
            torch.arange(1, w * self.stride + 1, self.stride)])
        p_0_x = torch.flatten(p_0_x).view(1, 1, h, w).repeat(1, N, 1, 1)
        p_0_y = torch.flatten(p_0_y).view(1, 1, h, w).repeat(1, N, 1, 1)
        p_0 = torch.cat([p_0_x, p_0_y], 1).type(dtype)
        # print('p_0 is ；',p_0,'\n','p_0 size is : ',p_0.size())
        # quit()
        p_0.requires_grad = False

        return p_0

    ####################################################################

    def _get_p(self, offset, dtype):
        N, h, w = offset.size(1) // 2, offset.size(2), offset.size(3)
        # print('***************************分隔符**************************')
        # print('offset.size(1) // 2, offset.size(2), offset.size(3), N,h,w is : ', N, h, w)

        # (1, 2N, 1, 1)
        p_n = self._get_p_n(N, dtype)  # p_n 是 一个确定的矩阵
        #  求的就是  《Deformable Convolutional Networks》中 公式1 或者公式2 中的 pn
        #  pn enumerates the locations in R.  R = {(−1,−1), (−1, 0), . . . , (0, 1), (1, 1)}
        #  - - pn_x - -     - - pn_y - -
        #   -1 -1  -1        -1  0   1
        #    0  0   0        -1  0   1           中间的就是 中心像素 p_0
        #    1  1   1        -1  0   1
        #  - - - - - -      - - - - - -
        # print('p_n is : ',p_n.size())
        '''p_n.size() is :  torch.Size([1, 18, 1, 1])
tensor([[[[-1.]],

         [[-1.]],

         [[-1.]],

         [[ 0.]],

         [[ 0.]],

         [[ 0.]],

         [[ 1.]],

         [[ 1.]],

         [[ 1.]],

         [[-1.]],

         [[ 0.]],

         [[ 1.]],

         [[-1.]],

         [[ 0.]],

         [[ 1.]],

         [[-1.]],

         [[ 0.]],

         [[ 1.]]]], device='cuda:0')
        '''
        # (1, 2N, h, w)
        p_0 = self._get_p_0(h, w, N, dtype)
        # p_0 是 一个确定的矩阵
        # For each location p_0 on the output feature map y   所以 p_0 就是输出图的 所有位置坐标
        '''
        p_0 size is :  torch.Size([1, 18, 2, 2])
        p_0 is ； 
     tensor([[[[1., 1.],
          [2., 2.]],
         [[1., 1.],
          [2., 2.]],
         [[1., 1.],
          [2., 2.]],
         [[1., 1.],
          [2., 2.]],
         [[1., 1.],
          [2., 2.]],
         [[1., 1.],
          [2., 2.]],
         [[1., 1.],
          [2., 2.]],
         [[1., 1.],
          [2., 2.]],
         [[1., 1.],
          [2., 2.]],
         [[1., 2.],
          [1., 2.]],
         [[1., 2.],
          [1., 2.]],
         [[1., 2.],
          [1., 2.]],
         [[1., 2.],
          [1., 2.]],
         [[1., 2.],
          [1., 2.]],
         [[1., 2.],
          [1., 2.]],
         [[1., 2.],
          [1., 2.]],
         [[1., 2.],
          [1., 2.]],
         [[1., 2.],
          [1., 2.]]]], device='cuda:0')
        '''
        # print('p_0 is : ',p_0.size())
        # print('***************************分隔符**************************')
        # quit()

        # print('p_0 size is : ', p_0.size())
        # print('p_n size is : ', p_n.size())
        # print('offset size is : ', offset.size())
        # quit()
        p_test_no_use = p_0 + p_n
        '''
        p_test_no_use is :
 tensor([[[[0., 0.],
          [1., 1.]],
         [[0., 0.],
          [1., 1.]],
         [[0., 0.],
          [1., 1.]],
         [[1., 1.],
          [2., 2.]],
         [[1., 1.],
          [2., 2.]],
         [[1., 1.],
          [2., 2.]],
         [[2., 2.],
          [3., 3.]],
         [[2., 2.],
          [3., 3.]],
         [[2., 2.],
          [3., 3.]],
         [[0., 1.],
          [0., 1.]],
         [[1., 2.],
          [1., 2.]],
         [[2., 3.],
          [2., 3.]],
         [[0., 1.],
          [0., 1.]],
         [[1., 2.],
          [1., 2.]],
         [[2., 3.],
          [2., 3.]],
         [[0., 1.],
          [0., 1.]],
         [[1., 2.],
          [1., 2.]],
         [[2., 3.],
          [2., 3.]]]], device='cuda:0')
        '''
        # print('p_test_no_use is : ','\n',p_test_no_use)
        p = p_0 + p_n + offset
        # print('p is :','\n',p)
        # quit()
        return p

    def _get_x_q(self, x, q, N):
        b, h, w, _ = q.size()
        padded_w = x.size(3)
        c = x.size(1)
        # (b, c, h*w)
        x = x.contiguous().view(b, c, -1)

        # (b, h, w, N)
        index = q[..., :N] * padded_w + q[..., N:]  # offset_x*w + offset_y
        # (b, c, h*w*N)
        index = index.contiguous().unsqueeze(dim=1).expand(-1, c, -1, -1, -1).contiguous().view(b, c, -1)
        '''
        >>> t = torch.Tensor([[1,2],[3,4]])
        >>> torch.gather(t,1,torch.LongTensor([[0,0],[1,0]])
            1  1
            4  3
        [torch.FloatTensor of size 2x2]
        首先，我们要看dim=1 or 0，这分别对应不同的维度进行操作，本例中dim=1表示在横向，所以索引就是列号，输出结果的大小与index的大小相同。
        index的第一行为[0,0]，第一个0索引到第一列即为1，第二个0也索引到第一列即为1；index的第二行为[1,0]，其中1索引到第二列即为4,0索引到第一列即为3，
        '''
        x_offset = x.gather(dim=-1, index=index).contiguous().view(b, c, h, w, N)  # dim=-1 估计是  最后一个 维度

        return x_offset

    @staticmethod
    def _reshape_x_offset(x_offset, ks):
        b, c, h, w, N = x_offset.size()
        x_offset = torch.cat([x_offset[..., s:s + ks].contiguous().view(b, c, h, w * ks) for s in range(0, N, ks)],
                             dim=-1)
        x_offset = x_offset.contiguous().view(b, c, h * ks, w * ks)

        return x_offset
class DeformConv2D_EPF1(nn.Module):

    def __init__(self, inc, outc, kernel_size=3, padding=1, stride=1):
        super(DeformConv2D_EPF1, self).__init__()

        self.conv = nn.Conv2d(inc, outc, kernel_size=3, stride=1, padding=1)

    # def forward(self, x):   # 大一圈
    #
    #     n_band = x.shape[1] - 1
    #     batch_size = x.shape[0]
    #     patch_size = x.shape[2]
    #     actual_patch_size = patch_size -2
    #     central_pixel = patch_size // 2
    #
    #     threshold_value = 8  # for PaviaU 0.5     减小  认为同类的 样本 变小
    #
    #     # print('run deform_conv, the input data size', x.size(),x.type())
    #
    #     x_numpy = x.cpu().numpy()
    #     # print("数组形状：", x_numpy.shape)  # 打印数组形状
    #
    #     # x_numpy[0, 103, :, :]  就是 传进来的 img_bilateralFilter_sum
    #     for i in range(batch_size):
    #
    #         # 将每个 batch_size 的输入数据的 img_bilateralFilter_sum 提取出来
    #         x_numpy_one = x_numpy[i, n_band, :, :]
    #         # print("数组形状：", x_numpy_one.shape)  # 打印数组形状
    #
    #         x_numpy_one_central_pixel_abs = abs(x_numpy_one - x_numpy_one[central_pixel,central_pixel])
    #         # print('x_numpy_one_central_pixel_abs:', x_numpy_one_central_pixel_abs)
    #
    #         x_numpy_one_central_pixel_abs_inter =  x_numpy_one_central_pixel_abs[1:patch_size-1,1:patch_size-1]
    #         x_numpy_one_central_pixel_abs_outsider = x_numpy_one_central_pixel_abs - np.pad(x_numpy_one_central_pixel_abs_inter, ((1,1), (1,1)), 'constant', constant_values=(0, 0))
    #
    #         # 得到跟中心像素不同类别的样本的坐标，就是将被替换掉像素的位置
    #         # 得到跟中心像素同类别的样本的坐标，就是不被替换掉像素的位置
    #         coordinates_notsame = np.argwhere(x_numpy_one_central_pixel_abs > threshold_value)
    #         coordinates_same = np.argwhere(x_numpy_one_central_pixel_abs < threshold_value)
    #         # print('\n','coordinates_notsame: ','\n',coordinates_notsame, '\n','coordinates_same: ','\n',coordinates_same)
    #
    #         # 在 x_numpy_one_central_pixel_abs_inter 得到跟中心像素不同类别的样本的坐标，就是将被替换掉像素的位置
    #         # 在 x_numpy_one_central_pixel_abs_inter得到跟中心像素同类别的样本的坐标，就是不被替换掉像素的位置
    #         coordinates_notsame_inter = np.argwhere(x_numpy_one_central_pixel_abs_inter > threshold_value)
    #         coordinates_same_inter = np.argwhere(x_numpy_one_central_pixel_abs_inter < threshold_value)
    #         # print('\n', 'coordinates_notsame_inter: ', '\n', coordinates_notsame_inter, '\n', 'coordinates_same_inter: ', '\n',
    #         #       coordinates_same_inter)
    #         # 在 x_numpy_one_central_pixel_abs_outsider 得到跟中心像素不同类别的样本的坐标，就是将被替换掉像素的位置
    #         # 在 x_numpy_one_central_pixel_abs_outsider 得到跟中心像素同类别的样本的坐标，就是不被替换掉像素的位置
    #         coordinates_notsame_outsider = np.argwhere(x_numpy_one_central_pixel_abs_outsider > threshold_value)
    #         coordinates_same_outsider = np.argwhere(x_numpy_one_central_pixel_abs_outsider < threshold_value)
    #         # print('\n', 'coordinates_notsame_outsider: ', '\n', coordinates_notsame_outsider, '\n', 'coordinates_same_outsider: ', '\n',
    #         #       coordinates_same_outsider)
    #
    #
    #         x_numpy_one_central_pixel_abs_outsider_flatten = x_numpy_one_central_pixel_abs_outsider.flatten()
    #         x_numpy_one_central_pixel_abs_outsider_flatten.sort()
    #         # print('\n', 'x_numpy_one_central_pixel_abs_outsider_flatten: ', '\n',
    #         #       x_numpy_one_central_pixel_abs_outsider_flatten)
    #
    #         if len(coordinates_notsame_inter) == 0 :
    #             # print('不需要可变卷积')
    #             pass
    #
    #         elif len(coordinates_notsame_inter) <= len(coordinates_same_outsider)-actual_patch_size * actual_patch_size:# actual_patch_size*actual_patch_size 内部需要替换的个数 小于等于 外部可以选择的个数
    #
    #             # print(i,'需要可变卷积','actual_patch_size*actual_patch_size 内部需要替换的个数 小于 外部可以选择的个数')
    #             # print('需要替换的像素点的个数是：', len(coordinates_notsame_inter))
    #             # # quit()
    #
    #             num_need_to_replace = len(coordinates_notsame_inter)
    #
    #             for m in range(num_need_to_replace):
    #                 coordinates1 = np.argwhere(x_numpy_one_central_pixel_abs == x_numpy_one_central_pixel_abs_outsider_flatten[m + actual_patch_size * actual_patch_size])
    #                 coordinates2 = coordinates_notsame_inter[m][0] + 1
    #                 coordinates3 = coordinates_notsame_inter[m][1] + 1
    #                 # print(coordinates1, m,coordinates2,coordinates3)
    #
    #                 x_numpy[i, :, coordinates2, coordinates3] = x_numpy[i, :, coordinates1[0][0],coordinates1[0][1]]
    #
    #         elif len(coordinates_notsame_inter) > len(coordinates_same_outsider)-actual_patch_size * actual_patch_size:  # actual_patch_size*actual_patch_size 内部需要替换的个数 大于 外部可以选择的个数
    #             #
    #             # print(i,'需要可变卷积', 'actual_patch_size*actual_patch_size 内部需要替换的个数 大于 外部可以选择的个数')
    #             # print('需要替换的像素点的个数是：', len(coordinates_notsame_inter),'可供选择的像素点个数：',len(coordinates_same_outsider)-actual_patch_size * actual_patch_size)
    #
    #             num_need_to_replace = len(coordinates_notsame_inter)
    #             num_can_to_replace = len(coordinates_same_outsider)-actual_patch_size * actual_patch_size
    #
    #             for m in range(num_need_to_replace):
    #                 # print('m',m)
    #
    #                 if m < num_can_to_replace:
    #                     coordinates1 = np.argwhere(x_numpy_one_central_pixel_abs == x_numpy_one_central_pixel_abs_outsider_flatten[ m + actual_patch_size * actual_patch_size])
    #                     coordinates2 = coordinates_notsame_inter[m][0] + 1
    #                     coordinates3 = coordinates_notsame_inter[m][1] + 1
    #                     # print(coordinates1, m, coordinates2, coordinates3)
    #                     x_numpy[i, :, coordinates2, coordinates3] = x_numpy[i, :, coordinates1[0][0],coordinates1[0][1]]
    #
    #                 if m >= num_can_to_replace:
    #                     coordinates2 = coordinates_notsame_inter[m][0] + 1
    #                     coordinates3 = coordinates_notsame_inter[m][1] + 1
    #                     # print(m, coordinates2, coordinates3)
    #                     x_numpy[i, :, coordinates2, coordinates3] = x_numpy[i, :, central_pixel,central_pixel]
    #
    #             # coordinates_notsame_x = np.zeros(coordinates_notsame.shape[0])
    #             # coordinates_notsame_y = np.zeros(coordinates_notsame.shape[0])
    #             #
    #             # print('\n','coordinates_notsame: ','\n',coordinates_notsame, '\n','coordinates_same: ','\n',coordinates_same)
    #             #
    #             # #计算位于 actual_patch_size*actual_patch_size 内的跟中心像素不同类的个数，
    #             # for j in range(len(coordinates_notsame)):
    #             #     if coordinates_notsame[j][0] in range(1,patch_size-1):
    #             #         if coordinates_notsame[j][1] in range(1, patch_size - 1):
    #             #
    #             #             coordinates_notsame_x[j] = coordinates_notsame[j][0]
    #             #             coordinates_notsame_y[j] = coordinates_notsame[j][1]
    #             #
    #             # print('\n','coordinates_notsame_x: ','\n',coordinates_notsame_x,sum(coordinates_notsame_x != 0))
    #             # print('\n', 'coordinates_notsame_y: ', '\n', coordinates_notsame_y, sum(coordinates_notsame_y != 0))
    #             # coordinates2 = coordinates_notsame_x[coordinates_notsame_x != 0].astype(int)
    #             # coordinates3 = coordinates_notsame_y[coordinates_notsame_y != 0].astype(int)
    #             #
    #
    #             # 不删，方法一   2020.10.5
    #             # for j in range(coordinates.shape[0]):
    #             #     # print(coordinates[j][0], coordinates[j][1])
    #             #
    #             #     x_numpy[i, :, coordinates[j][0], coordinates[j][1]] = x_numpy[i, :, central_pixel, central_pixel]
    #             #
    #
    #     x_offset = x_numpy[:, 0:n_band, 1:patch_size-1,1:patch_size-1]
    #
    #     # print(x_offset.shape)
    #     x_offset = torch.tensor(x_offset).cuda()
    #     # x_offset.to(cuda)
    #     # print(x_offset.type())
    #     # quit()
    #
    #         #
    #         # if each_class_number[0] > actual_patch_size*actual_patch_size:   #代表 传进来的 patch_size*patch_size 中 包含足够的 actual_patch_size*actual_patch_size，用于卷积
    #         #
    #         #     coordinates = np.argwhere(x_numpy_one_central_pixel_abs_threshold_value0 == 0)
    #         #     print(coordinates)
    #         #     for j in range(each_class_number[0]):
    #         #         print(coordinates[j][0])
    #         #         if coordinates[j][0] in range(1,patch_size-1):
    #         #
    #         #             print(coordinates[j])
    #         #
    #         #
    #         #
    #         #
    #         #     quit()
    #         #
    #         #     x_offset = np.zeros((n_band,actual_patch_size, actual_patch_size))#, dtype='')
    #         #     print(x_offset.shape)
    #         #
    #         #
    #         #
    #         #
    #
    #     out = self.conv(x_offset)
    #     #print('out size is : ', out.size())
    #     #quit()
    #     return out
    def forward(self, x):   # 大两圈

        big_size = 2

        n_band = x.shape[1] - 1
        batch_size = x.shape[0]
        patch_size = x.shape[2]
        actual_patch_size = patch_size - 2*big_size
        central_pixel = patch_size // 2

        threshold_value = 0.5  # for PaviaU 0.5     减小  认为同类的 样本 变小

        # print('run deform_conv, the input data size', x.size(),x.type())

        x_numpy = x.cpu().numpy()
        # print("数组形状：", x_numpy.shape)  # 打印数组形状

        # x_numpy[0, 103, :, :]  就是 传进来的 img_bilateralFilter_sum
        for i in range(batch_size):

            # 将每个 batch_size 的输入数据的 img_bilateralFilter_sum 提取出来
            x_numpy_one = x_numpy[i, n_band, :, :]
            print("数组形状：", x_numpy_one.shape)  # 打印数组形状

            x_numpy_one_central_pixel_abs = abs(x_numpy_one - x_numpy_one[central_pixel,central_pixel])
            # print('x_numpy_one_central_pixel_abs:', x_numpy_one_central_pixel_abs)

            x_numpy_one_central_pixel_abs_inter =  x_numpy_one_central_pixel_abs[big_size:patch_size-big_size,big_size:patch_size-big_size]
            x_numpy_one_central_pixel_abs_outsider = x_numpy_one_central_pixel_abs - np.pad(x_numpy_one_central_pixel_abs_inter, ((big_size,big_size), (big_size,big_size)), 'constant', constant_values=(0, 0))

            # 得到跟中心像素不同类别的样本的坐标，就是将被替换掉像素的位置
            # 得到跟中心像素同类别的样本的坐标，就是不被替换掉像素的位置
            coordinates_notsame = np.argwhere(x_numpy_one_central_pixel_abs > threshold_value)
            coordinates_same = np.argwhere(x_numpy_one_central_pixel_abs < threshold_value)
            # print('\n','coordinates_notsame: ','\n',coordinates_notsame, '\n','coordinates_same: ','\n',coordinates_same)

            # 在 x_numpy_one_central_pixel_abs_inter 得到跟中心像素不同类别的样本的坐标，就是将被替换掉像素的位置
            # 在 x_numpy_one_central_pixel_abs_inter得到跟中心像素同类别的样本的坐标，就是不被替换掉像素的位置
            coordinates_notsame_inter = np.argwhere(x_numpy_one_central_pixel_abs_inter > threshold_value)
            coordinates_same_inter = np.argwhere(x_numpy_one_central_pixel_abs_inter < threshold_value)
            # print('\n', 'coordinates_notsame_inter: ', '\n', coordinates_notsame_inter, '\n', 'coordinates_same_inter: ', '\n',
            #       coordinates_same_inter)
            # 在 x_numpy_one_central_pixel_abs_outsider 得到跟中心像素不同类别的样本的坐标，就是将被替换掉像素的位置
            # 在 x_numpy_one_central_pixel_abs_outsider 得到跟中心像素同类别的样本的坐标，就是不被替换掉像素的位置
            coordinates_notsame_outsider = np.argwhere(x_numpy_one_central_pixel_abs_outsider > threshold_value)
            coordinates_same_outsider = np.argwhere(x_numpy_one_central_pixel_abs_outsider < threshold_value)
            # print('\n', 'coordinates_notsame_outsider: ', '\n', coordinates_notsame_outsider, '\n', 'coordinates_same_outsider: ', '\n',
            #       coordinates_same_outsider)

            x_numpy_one_central_pixel_abs_outsider_flatten = x_numpy_one_central_pixel_abs_outsider.flatten()
            x_numpy_one_central_pixel_abs_outsider_flatten.sort()
            # print('\n', 'x_numpy_one_central_pixel_abs_outsider_flatten: ', '\n',
            #       x_numpy_one_central_pixel_abs_outsider_flatten)

            if len(coordinates_notsame_inter) == 0 :
                # print('不需要可变卷积')
                pass

            elif len(coordinates_notsame_inter) <= len(coordinates_same_outsider)-actual_patch_size * actual_patch_size:# actual_patch_size*actual_patch_size 内部需要替换的个数 小于等于 外部可以选择的个数

                # print(i,'需要可变卷积','actual_patch_size*actual_patch_size 内部需要替换的个数 小于 外部可以选择的个数')
                # print('需要替换的像素点的个数是：', len(coordinates_notsame_inter))
                # # quit()

                num_need_to_replace = len(coordinates_notsame_inter)

                for m in range(num_need_to_replace):
                    coordinates1 = np.argwhere(x_numpy_one_central_pixel_abs == x_numpy_one_central_pixel_abs_outsider_flatten[m + actual_patch_size * actual_patch_size])
                    coordinates2 = coordinates_notsame_inter[m][0] + 1
                    coordinates3 = coordinates_notsame_inter[m][1] + 1
                    # print(coordinates1, m,coordinates2,coordinates3)

                    x_numpy[i, :, coordinates2, coordinates3] = x_numpy[i, :, coordinates1[0][0],coordinates1[0][1]]

            elif len(coordinates_notsame_inter) > len(coordinates_same_outsider)-actual_patch_size * actual_patch_size:  # actual_patch_size*actual_patch_size 内部需要替换的个数 大于 外部可以选择的个数
                #
                # print(i,'需要可变卷积', 'actual_patch_size*actual_patch_size 内部需要替换的个数 大于 外部可以选择的个数')
                # print('需要替换的像素点的个数是：', len(coordinates_notsame_inter),'可供选择的像素点个数：',len(coordinates_same_outsider)-actual_patch_size * actual_patch_size)

                num_need_to_replace = len(coordinates_notsame_inter)
                num_can_to_replace = len(coordinates_same_outsider)-actual_patch_size * actual_patch_size

                for m in range(num_need_to_replace):
                    # print('m',m)

                    if m < num_can_to_replace:
                        coordinates1 = np.argwhere(x_numpy_one_central_pixel_abs == x_numpy_one_central_pixel_abs_outsider_flatten[ m + actual_patch_size * actual_patch_size])
                        coordinates2 = coordinates_notsame_inter[m][0] + 1
                        coordinates3 = coordinates_notsame_inter[m][1] + 1
                        # print(coordinates1, m, coordinates2, coordinates3)
                        x_numpy[i, :, coordinates2, coordinates3] = x_numpy[i, :, coordinates1[0][0],coordinates1[0][1]]

                    if m >= num_can_to_replace:
                        coordinates2 = coordinates_notsame_inter[m][0] + 1
                        coordinates3 = coordinates_notsame_inter[m][1] + 1
                        # print(m, coordinates2, coordinates3)
                        x_numpy[i, :, coordinates2, coordinates3] = x_numpy[i, :, central_pixel,central_pixel]

                # coordinates_notsame_x = np.zeros(coordinates_notsame.shape[0])
                # coordinates_notsame_y = np.zeros(coordinates_notsame.shape[0])
                #
                # print('\n','coordinates_notsame: ','\n',coordinates_notsame, '\n','coordinates_same: ','\n',coordinates_same)
                #
                # #计算位于 actual_patch_size*actual_patch_size 内的跟中心像素不同类的个数，
                # for j in range(len(coordinates_notsame)):
                #     if coordinates_notsame[j][0] in range(1,patch_size-1):
                #         if coordinates_notsame[j][1] in range(1, patch_size - 1):
                #
                #             coordinates_notsame_x[j] = coordinates_notsame[j][0]
                #             coordinates_notsame_y[j] = coordinates_notsame[j][1]
                #
                # print('\n','coordinates_notsame_x: ','\n',coordinates_notsame_x,sum(coordinates_notsame_x != 0))
                # print('\n', 'coordinates_notsame_y: ', '\n', coordinates_notsame_y, sum(coordinates_notsame_y != 0))
                # coordinates2 = coordinates_notsame_x[coordinates_notsame_x != 0].astype(int)
                # coordinates3 = coordinates_notsame_y[coordinates_notsame_y != 0].astype(int)
                #

                # 不删，方法一   2020.10.5
                # for j in range(coordinates.shape[0]):
                #     # print(coordinates[j][0], coordinates[j][1])
                #
                #     x_numpy[i, :, coordinates[j][0], coordinates[j][1]] = x_numpy[i, :, central_pixel, central_pixel]
                #

        x_offset = x_numpy[:, 0:n_band, big_size:patch_size-big_size,big_size:patch_size-big_size]

        # print(x_offset.shape)
        x_offset = torch.tensor(x_offset).cuda()
        # x_offset.to(cuda)
        # print(x_offset.type())
        # quit()

            #
            # if each_class_number[0] > actual_patch_size*actual_patch_size:   #代表 传进来的 patch_size*patch_size 中 包含足够的 actual_patch_size*actual_patch_size，用于卷积
            #
            #     coordinates = np.argwhere(x_numpy_one_central_pixel_abs_threshold_value0 == 0)
            #     print(coordinates)
            #     for j in range(each_class_number[0]):
            #         print(coordinates[j][0])
            #         if coordinates[j][0] in range(1,patch_size-1):
            #
            #             print(coordinates[j])
            #
            #
            #
            #
            #     quit()
            #
            #     x_offset = np.zeros((n_band,actual_patch_size, actual_patch_size))#, dtype='')
            #     print(x_offset.shape)
            #
            #
            #
            #

        out = self.conv(x_offset)
        #print('out size is : ', out.size())
        #quit()
        return out
class DeformConv2D_EPF2(nn.Module):

    def __init__(self, inc, outc, kernel_size=3, padding=1, stride=1):
        super(DeformConv2D_EPF2, self).__init__()

        self.conv = nn.Conv2d(inc, outc, kernel_size=3, stride=1, padding=1)

        # self.weight_1 = torch.nn.Parameter(torch.FloatTensor(1).cuda(), requires_grad=True)
        # self.weight_1.data.fill_(0.1)
        # self.weight_2 = torch.nn.Parameter(torch.FloatTensor(1).cuda(), requires_grad=True)
        # self.weight_2.data.fill_(10)

    def forward(self, x):

        y = x
        def bilateral_filter1(image, gsigma, ssigma):
            img = image.detach()
            # img = image
            batch_size = img.shape[0]
            n_band = img.shape[1]
            patch_size = img.shape[2]
            real_patchsize = patch_size - 6
            img = img.permute(0, 2, 3, 1)

            gkernel = gaus_kernel(patch_size, gsigma).cuda()
            gkernel = gkernel.repeat(batch_size, 1, 1)
            sigma2 = 2 * torch.mul(ssigma, ssigma).float()
            imgg = torch.zeros([batch_size, patch_size, patch_size, n_band]).cuda()

            for i in range(batch_size):
                image_central = (img[i, :, :, :] - img[i, patch_size // 2, patch_size // 2, :]).cuda()
                imgg[i, :, :, :] = image_central

            image_central = torch.exp(-(imgg * imgg) / sigma2).cuda()
            image_central = torch.sum(image_central, 3) / n_band

            # image_central[image_central > 0.7] = 1
            # image_central[image_central < 0.5] = 0
            #
            # aaa = image_central[1, :, :].cpu().detach().numpy()
            # prediction_data2 = pd.DataFrame(aaa)
            # data2 = pd.ExcelWriter('image_central.xlsx')  # 写入Excel文件
            # prediction_data2.to_excel(data2, 'image_central', float_format='%.5f')  # ‘page_1’是写入excel的sheet名
            # data2.save()

            gkernel[image_central > 0.9] = 1
            image_central = image_central * gkernel
            # image_central[image_central > 0.9] = 1
            # image_central[image_central < 0.2] = 0
            # aa = image_central[0, :, :].cpu().detach().numpy()
            # prediction_data2 = pd.DataFrame(aa)
            # data2 = pd.ExcelWriter('image_central1.xlsx')  # 写入Excel文件
            # prediction_data2.to_excel(data2, 'image_central', float_format='%.5f')  # ‘page_1’是写入excel的sheet名
            # data2.save()
            # quit()

            # for i in range(batch_size):
            #
            #     image_central_one = image_central[i, :, :]
            #     a = image_central_one.view(-1)
            #     aa, _ = torch.sort(a, descending=True) # 从大 到 小 排列
            #
            #
            #     # image_central_one[image_central_one > aa[real_patchsize*real_patchsize - 1]] = 1
            #     image_central_one[image_central_one < aa[real_patchsize * real_patchsize - 1]] = 0
            #     # image_central_one[image_central_one < 0.1] = 0
            #     # image_central_one[image_central_one >= 0.1] = 1
            #     image_central[i, :, :] = image_central_one

            # aa = image_central[0, :, :].cpu().detach().numpy()
            # prediction_data2 = pd.DataFrame(aa)
            # data2 = pd.ExcelWriter('image_central11.xlsx')  # 写入Excel文件
            # prediction_data2.to_excel(data2, 'image_central', float_format='%.5f')  # ‘page_1’是写入excel的sheet名
            # data2.save()
            # quit()

            # from scipy.spatial import distance
            # offset = torch.zeros(batch_size,patch_size,patch_size,2).cuda()
            # # print(offset.shape)
            # # quit()
            # coo_not_one_b, coo_not_one_x, coo_not_one_y = torch.where(image_central == 0.)
            # coo_one_b, coo_one_x, coo_one_y = torch.where(image_central == 1.)
            # # print(coo_not_one_b,'\n', coo_not_one_x,'\n', coo_not_one_y)
            # # quit()
            # print(coo_not_one_b.shape)
            # num = np.array([(torch.where(coo_not_one_b == g)[0]).numel() for g in range(batch_size)])
            # print(num, num.shape, num.sum())
            # num1 = np.array([(torch.where(coo_one_b == g)[0]).numel() for g in range(batch_size)])
            # print(num1, num1.shape, num1.sum())
            # # quit()
            # num_chazhi = np.where((num-num1) > 0, num1, num)
            # print(num_chazhi)
            # # quit()
            #
            # coo = np.array(
            #     [(coo_not_one_b[a], coo_not_one_x[a], coo_not_one_y[a]) for a in range(coo_not_one_b.shape[0])])
            # coo1 = np.array(
            #     [(coo_one_b[a], coo_one_x[a], coo_one_y[a]) for a in range(coo_one_b.shape[0])])
            # print(coo)
            #
            #
            #
            # quit()

            image_central = image_central.unsqueeze(3)
            imgg = image_central.repeat(1, 1, 1, n_band)
            # print(imgg.shape,img.shape)
            # quit()
            imgg = imgg.permute(0, 3, 1, 2)
            return imgg

        x_weighting = bilateral_filter1(y, 10, 0.1).cuda()

        # x_weighting[x_weighting > 0.9] = 1
        # x_weighting[x_weighting < self.weight_1] = 0

        # x_offset = x * x_weighting
        x[x_weighting < 0.2] = 0
        # print(self.weight_1)
        # print(self.weight_2)
        # print(x_weighting.shape)
        # quit()
        # #
        # patch_size = x.shape[2]
        # real_patchsize = patch_size - 6
        # x_offset = x_offset.permute(0, 2, 3, 1)
        # x_offset_sum = torch.sum(x_offset, 3)
        # x_offset1 = x_offset[x_offset_sum != 0]
        # print(x_offset1.shape)
        # quit()
        # x_offset1 = x_offset1[0:x.shape[0]*real_patchsize*real_patchsize,:]
        # x_offset = x_offset1.view(x.shape[0],real_patchsize,real_patchsize,x.shape[1])
        # x_offset = x_offset.permute(0, 3, 1, 2)

        out = self.conv(x)

        return out
# class DeformConv2D_EPF(nn.Module):
#
#     def __init__(self, inc, outc, kernel_size=3, padding=1, stride=1):
#         super(DeformConv2D_EPF, self).__init__()
#
#         self.conv = nn.Conv2d(inc, outc, kernel_size=3, stride=1, padding=1)
#
#     def forward(self, x):
#
#         batch_size = x.shape[0]
#         n_band     = x.shape[1]
#         patch_size = x.shape[2]
#         real_patchsize = patch_size -2
#         central_pixel = patch_size // 2
#         central_pixel_one =  patch_size*patch_size // 2
#         # print('run deform_conv, the input data size', x.size(),x.type(),central_pixel,central_pixel_one)
#
#
#         x_numpy = x
#         x_numpy = x_numpy.cpu().numpy()
#         # print("数组形状：", x_numpy.shape)  # 打印数组形状
#
#         # x_numpy[0, 103, :, :]  就是 传进来的 img_bilateralFilter_sum
#         for i in range(batch_size):
#
#             # 将每个 batch_size 的输入数据的 img_bilateralFilter_sum 提取出来
#             x_numpy_one = x_numpy[i, :, :, :]
#             # print(x_numpy_one.shape)
#             x_numpy_one = x_numpy_one.transpose(1, 2, 0)
#             # print("数组形状：", x_numpy_one.shape)  # 打印数组形状
#             # print("hhhhhhhhhhhhhhhhh")
#             _,img_bilateralFilter = bilateralFilter_img(x_numpy_one, 5, 100, 100)
#             # print("数组形状：", img_bilateralFilter.shape)  # 打印数组形状
#             # print(img_bilateralFilter,img_bilateralFilter[2][2],img_bilateralFilter[1][0])
#             # print("hhhhhhhhhhhhhhhhh")
#
#             img_bilateralFilter_reshape = np.reshape(img_bilateralFilter, (-1, 3))
#             # print("数组形状：", img_bilateralFilter_reshape.shape,img_bilateralFilter)  # 打印数组形状
#
#
#             # print("hhhhhhhhhhhhdddddhhhhh")
#             img_bilateralFilter_reshape_cov = np.cov(img_bilateralFilter_reshape)
#             # print(img_bilateralFilter_reshape_cov.shape,img_bilateralFilter_reshape_cov)
#
#             # print("hhhhhhhhhhhhhhhhh")
#             img_bilateralFilter_cov = img_bilateralFilter_reshape_cov[central_pixel_one,:]
#             # print(img_bilateralFilter_cov)
#             img_bilateralFilter_cov = img_bilateralFilter_cov.reshape([patch_size,patch_size])
#             print(img_bilateralFilter_cov)
#             coordinates = np.argwhere(img_bilateralFilter_cov < 0)
#             print(coordinates,len(coordinates))
#
#             # quit()
#             mod = 1  # 选择 模式  mod = 1 代表选择 patch_size*patch_size 内同类的样本，剔除其他的样本，并用中心位置光谱值代替
#                      #           mod = 2 代表选择 patch_size*patch_size 中 内同类的样本，剔除 real_patchsize*real_patchsize 中其他
#
#             # if len(coordinates) == 0:
#             #     print("不需要可变卷积")
#             #     pass
#             # else:
#             #     for j in range(len(coordinates)):
#             #         # print(coordinates[j][0], coordinates[j][1])
#             #         x_numpy[i, :, coordinates[j][0], coordinates[j][1]] = x_numpy[i, :, central_pixel,
#             #                                                                   central_pixel]
#
#             # quit()
#
#
#             for j in range(len(coordinates)):
#
#                 print(coordinates[j])
#
#                 # if coordinates[j][0] in range(1,patch_size-1):
#                 #
#                 #
#                 #     print(coordinates[j])
#
#             quit()
#
#
#
#
#
#         x_offset = x_numpy
#         # print(x_offset.shape)
#         x_offset = torch.tensor(x_offset).cuda()
#         # x_offset.to(cuda)
#         # quit()
#
#             #
#             # if each_class_number[0] > actual_patch_size*actual_patch_size:   #代表 传进来的 patch_size*patch_size 中 包含足够的 actual_patch_size*actual_patch_size，用于卷积
#             #
#             #     coordinates = np.argwhere(x_numpy_one_central_pixel_abs_threshold_value0 == 0)
#             #     print(coordinates)
#             #     for j in range(each_class_number[0]):
#             #         print(coordinates[j][0])
#             #         if coordinates[j][0] in range(1,patch_size-1):
#             #
#             #             print(coordinates[j])
#             #
#             #
#             #
#             #
#             #     quit()
#             #
#             #     x_offset = np.zeros((n_band,actual_patch_size, actual_patch_size))#, dtype='')
#             #     print(x_offset.shape)
#             #
#             #
#             #
#             #
#
#         out = self.conv(x_offset)
#         # print('out size is : ', out.size())
#         quit()
#         return out
class DeformConv2D_EPF3(nn.Module):

    def __init__(self, inc, outc, kernel_size, bias=None,):
        super(DeformConv2D_EPF3, self).__init__()

        self.conv = nn.Conv2d(inc, outc, kernel_size=kernel_size, stride=kernel_size, bias=bias)

    def forward(self, x):
        y = x
        def bilateral_filter1(image, gsigma, ssigma):
            kernel_size = 7
            biger_size = kernel_size

            img = image.detach()
            batch_size = img.shape[0]
            n_band = img.shape[1]
            patch_size = img.shape[2]
            expand_patch_size = (patch_size - kernel_size + 1) * kernel_size

            img = img.permute(0, 2, 3, 1)

            gkernel = gaus_kernel(patch_size, gsigma).cuda()
            gkernel = gkernel.repeat(batch_size, 1, 1)
            sigma2 = 2 * torch.mul(ssigma, ssigma).float()
            imgg = torch.zeros([batch_size, patch_size, patch_size, n_band])
            imgg1 = torch.zeros([batch_size, expand_patch_size, expand_patch_size, n_band])

            for i in range(batch_size):
                image_central = (img[i, :, :, :] - img[i, patch_size // 2, patch_size // 2, :]).cuda()
                imgg[i, :, :, :] = image_central

            image_central = torch.exp(-(imgg * imgg) / sigma2).cuda()
            image_central = torch.sum(image_central, 3) / n_band

            # image_central[image_central > 0.7] = 1
            # image_central[image_central < 0.5] = 0
            #

            gkernel[image_central > 0.9] = 1
            image_central = image_central * gkernel
            image_central[image_central > 0.8] = 1
            print(image_central.size())

            # aaa = image_central[1, :, :].cpu().detach().numpy()
            # prediction_data2 = pd.DataFrame(aaa)
            # data2 = pd.ExcelWriter('image_central.xlsx')  # 写入Excel文件
            # prediction_data2.to_excel(data2, 'image_central', float_format='%.5f')  # ‘page_1’是写入excel的sheet名
            # data2.save()
            # quit()

            for i in range(batch_size):
                # print(i)
                image_central_one = image_central[i, :, :]
                # print(image_central_one.size())
                for m in range(patch_size - kernel_size + 1):
                    for n in range(patch_size - kernel_size + 1):
                        # m,n = 10,5
                        # print(m,n)
                        # print(image_central_one[m:m + kernel_size, n:n + kernel_size])

                        coo_one_x,coo_one_y = torch.where(image_central_one[m:m + kernel_size, n:n + kernel_size] == 1.)
                        real_coo_one_x, real_coo_one_y = m + coo_one_x,n + coo_one_y
                        coo_not_one_x, coo_not_one_y = torch.where(image_central_one[m:m + kernel_size, n:n + kernel_size] != 1.)
                        real_coo_not_one_x, real_coo_not_one_y = m + coo_not_one_x, n + coo_not_one_y

                        # print(coo_one_x,coo_one_y,real_coo_one_x, real_coo_one_y)
                        # print(len(coo_one_x))
                        # quit()

                        if len(coo_one_x) != kernel_size * kernel_size:
                            # print('len(coo_not_one_x) != kernel_size * kernel_size')

                            def calculate_num_the_same_class(biger_size):
                                # print('biger_size',biger_size)

                                x1, y1 = m,n
                                x2, y2 = x1 + kernel_size, y1 + kernel_size

                                # print('x1,x2, y1,y2',x1,x2, y1,y2)

                                x_begin = x1 - biger_size
                                x_end = x2 + biger_size
                                y_begin = y1 - biger_size
                                y_end = y2 + biger_size
                                if x_begin >= 0 and x_end <= patch_size and y_begin >= 0 and y_end <= patch_size:
                                    pass
                                else:
                                    if x_begin < 0:
                                        x_begin = 0
                                    if x_end > patch_size:
                                        x_end = patch_size
                                    if y_begin < 0:
                                        y_begin = 0
                                    if y_end > patch_size:
                                        y_end = patch_size

                                # print('x_begin:x_end, y_begin:y_end',x_begin,x_end, y_begin,y_end)

                                image_central_one_out = image_central_one[x_begin:x_end, y_begin:y_end]
                                # print('输入 image_central_one_out 数据：', image_central_one_out)

                                pad = nn.ZeroPad2d(padding=(y1 - y_begin, y_end - y2, x1 - x_begin, x_end - x2))

                                yy = pad(image_central_one[m:m + kernel_size, n:n + kernel_size])

                                image_central_one_out1 = image_central_one_out - yy

                                coo_out_one_x, coo_out_one_y = torch.where(image_central_one_out1 == 1.)

                                # print('\n', 'coo_out_one_x: ', '\n', coo_out_one_x, '\n', 'coo_out_one_y: ', '\n',coo_out_one_y)
                                # quit()

                                if biger_size <= patch_size-1:
                                    if len(coo_out_one_x) >= len(coo_not_one_x):

                                        real_coo_out_one_x = coo_out_one_x + x_begin
                                        real_coo_out_one_y = coo_out_one_y + y_begin

                                        return real_coo_out_one_x, real_coo_out_one_y
                                    else:
                                        return calculate_num_the_same_class(biger_size + 1)
                                else:
                                    real_coo_out_one_x = coo_out_one_x + x_begin
                                    real_coo_out_one_y = coo_out_one_y + y_begin
                                    return real_coo_out_one_x, real_coo_out_one_y

                            real_coo_out_one_x1, real_coo_out_one_y1 = calculate_num_the_same_class(biger_size)

                            # print('\n', 'real_coo_out_one_x: ', '\n', real_coo_out_one_x1, '\n', 'real_coo_out_one_y: ', '\n',real_coo_out_one_y1)

                            for k in range(min(len(coo_not_one_x),len(real_coo_out_one_x1))):
                                img_one = img
                                # print(real_coo_not_one_x, real_coo_not_one_y)
                                # print(real_coo_not_one_x[k].item(),real_coo_not_one_y[k].item())
                                # print(real_coo_out_one_x1[k].item(),real_coo_out_one_y1[k].item())
                                img_one[i,real_coo_not_one_x[k].item(),real_coo_not_one_y[k].item(),:] = \
                                    img[i,real_coo_out_one_x1[k].item(),real_coo_out_one_y1[k].item(),:]

                                imgg1[i, m * kernel_size:m * kernel_size + kernel_size, n * kernel_size:n * kernel_size + kernel_size, :] = \
                                    img_one[i, m:m + kernel_size, n:n + kernel_size, :]
                        else:
                            # print('不需要可变卷积')
                            imgg1[i, m * kernel_size:m * kernel_size + kernel_size,n * kernel_size:n * kernel_size + kernel_size, :] = \
                                img[i, m:m + kernel_size, n:n + kernel_size, :]

            return imgg1

            # image_central[image_central < 0.1] = 0
            # aa = image_central[0, :, :].cpu().detach().numpy()
            # prediction_data2 = pd.DataFrame(aa)
            # data2 = pd.ExcelWriter('image_central1.xlsx')  # 写入Excel文件
            # prediction_data2.to_excel(data2, 'image_central', float_format='%.5f')  # ‘page_1’是写入excel的sheet名
            # data2.save()
            # quit()

            # for i in range(batch_size):
            #     image_central_one = image_central[i, :, :]
            #     a = image_central_one.view(-1)
            #     aa, _ = torch.sort(a, descending=True) # 从大 到 小 排列
            #
            #     # image_central_one[image_central_one > aa[real_patchsize*real_patchsize - 1]] = 1
            #     image_central_one[image_central_one < aa[real_patchsize * real_patchsize - 1]] = 0
            #     image_central[i, :, :] = image_central_one

            # aa = image_central[0, :, :].cpu().detach().numpy()
            # prediction_data2 = pd.DataFrame(aa)
            # data2 = pd.ExcelWriter('image_central11.xlsx')  # 写入Excel文件
            # prediction_data2.to_excel(data2, 'image_central', float_format='%.5f')  # ‘page_1’是写入excel的sheet名
            # data2.save()
            # quit()



            # image_central = image_central.unsqueeze(3)
            # imgg = image_central.repeat(1, 1, 1, n_band)
            # imgg = imgg.permute(0, 3, 1, 2)
            # return imgg

        x_offset = bilateral_filter1(y, 10, 0.1).cuda()
        x_offset = x_offset.permute(0, 3, 1, 2)
        out = self.conv(x_offset)
        # print('out size is : ', out.size())
        # quit()
        return out
class DeformConv2D_EPF4(nn.Module):

    def __init__(self, inc, outc, kernel_size=3, padding=1, stride=1):
        super(DeformConv2D_EPF4, self).__init__()

        self.conv = nn.Conv2d(inc, outc, kernel_size=7, stride=1, padding=1)

    def forward(self, x):

        y = x

        def bilateral_filter1(image, gsigma, ssigma):
            img = image.detach()
            batch_size = img.shape[0]
            n_band = img.shape[1]
            patch_size = img.shape[2]
            real_patchsize = patch_size - 6
            img = img.permute(0, 2, 3, 1)

            gkernel = gaus_kernel(patch_size, gsigma).cuda()
            gkernel = gkernel.repeat(batch_size, 1, 1)
            sigma2 = 2 * torch.mul(ssigma, ssigma).float()
            imgg = torch.zeros([batch_size, patch_size, patch_size, n_band])

            for i in range(batch_size):
                image_central = (img[i, :, :, :] - img[i, patch_size // 2, patch_size // 2, :]).cuda()
                imgg[i, :, :, :] = image_central

            image_central = torch.exp(-(imgg * imgg) / sigma2).cuda()
            image_central = torch.sum(image_central, 3) / n_band

            # image_central[image_central > 0.7] = 1
            # image_central[image_central < 0.5] = 0
            #
            # aaa = image_central[1, :, :].cpu().detach().numpy()
            # prediction_data2 = pd.DataFrame(aaa)
            # data2 = pd.ExcelWriter('image_central.xlsx')  # 写入Excel文件
            # prediction_data2.to_excel(data2, 'image_central', float_format='%.5f')  # ‘page_1’是写入excel的sheet名
            # data2.save()

            gkernel[image_central > 0.9] = 1
            image_central = image_central * gkernel
            # image_central[image_central > 0.8] = 1
            image_central[image_central <= 0.5] = 0
            # aa = image_central[0, :, :].cpu().detach().numpy()
            # prediction_data2 = pd.DataFrame(aa)
            # data2 = pd.ExcelWriter('image_central1.xlsx')  # 写入Excel文件
            # prediction_data2.to_excel(data2, 'image_central', float_format='%.5f')  # ‘page_1’是写入excel的sheet名
            # data2.save()


            # aa = image_central[0, :, :].cpu().detach().numpy()
            # prediction_data2 = pd.DataFrame(aa)
            # data2 = pd.ExcelWriter('image_central11.xlsx')  # 写入Excel文件
            # prediction_data2.to_excel(data2, 'image_central', float_format='%.5f')  # ‘page_1’是写入excel的sheet名
            # data2.save()
            # quit()
            image_central = image_central.unsqueeze(3)
            imgg = image_central.repeat(1, 1, 1, n_band)
            imgg = imgg.permute(0, 3, 1, 2)
            return imgg

        x_weighting = bilateral_filter1(y, 10, 0.1).cuda()

        # x_weighting[x_weighting > 0.9] = 1
        # x_weighting[x_weighting < 0.1] = 0

        x_offset = x * x_weighting
        #
        for i in range(x_offset.shape[0]):
            x_offset_one = x_offset[i,:,:,:]
            x_offset_one = x_offset_one.permute(1, 2, 0)
            patch_size = x_offset_one.shape[1]
            # print(x_offset_one.shape,patch_size)

            x_offset_sum = torch.sum(x_offset_one, 2)
            # print(x_offset_sum)
            #
            #
            # prediction_data2 = pd.DataFrame(x_offset_sum.cpu().numpy())
            # data2 = pd.ExcelWriter('image_central11.xlsx')  # 写入Excel文件
            # prediction_data2.to_excel(data2, 'image_central', float_format='%.5f')  # ‘page_1’是写入excel的sheet名
            # data2.save()

            x_offset_one1 = x_offset_one[x_offset_sum != 0]
            # real_patchsize = int(math.sqrt(x_offset_one1.shape[0]))

            # x_offset_one1 = x_offset_one1[0:real_patchsize * real_patchsize, :]
            # print(x_offset_one1.shape)
            padzero = torch.zeros((patch_size * patch_size - x_offset_one1.shape[0]), x_offset_one.shape[2]).cuda()
            x_offset_one1 = torch.cat((x_offset_one1, padzero),0)
            # print(x_offset_one1.shape)
            # quit()

            x_offset_one1 = x_offset_one1.view(patch_size, patch_size, x_offset_one.shape[2])
            # print(x_offset_one1.shape)
            #
            # pad = nn.ZeroPad2d((patch_size - real_patchsize))
            #
            # x_offset_one1 = pad(x_offset_one1)
            # print(x_offset_one1.shape)
            # quit()

            x_offset_one1 = x_offset_one1.permute(2,0,1)
            # print(x_offset_one1.shape)

            # quit()
            x_offset[i, :, :, :] = x_offset_one1

        out = self.conv(x_offset)

        return out
class DeformConv2D_EPF5(nn.Module):

    def __init__(self, inc, outc, kernel_size=3, padding=1, stride=1):
        super(DeformConv2D_EPF5, self).__init__()

        self.conv = nn.Conv2d(inc, outc, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        y = x
        def bilateral_filter1(image, gsigma, ssigma):

            # kernel_size = 3
            # biger_size = kernel_size
            img = image.detach()
            batch_size = img.shape[0]
            n_band = img.shape[1]
            patch_size = img.shape[2]
            # expand_patch_size = (patch_size - kernel_size + 1) * kernel_size

            img = img.permute(0, 2, 3, 1)

            gkernel = gaus_kernel(patch_size, gsigma).cuda()
            gkernel = gkernel.repeat(batch_size, 1, 1)
            sigma2 = 2 * torch.mul(ssigma, ssigma).float()
            imgg = torch.zeros([batch_size, patch_size, patch_size, n_band])
            # imgg1 = torch.zeros([batch_size, expand_patch_size, expand_patch_size, n_band])

            for i in range(batch_size):
                image_central = (img[i, :, :, :] - img[i, patch_size // 2, patch_size // 2, :]).cuda()
                imgg[i, :, :, :] = image_central

            image_central = torch.exp(-(imgg * imgg) / sigma2).cuda()
            image_central = torch.sum(image_central, 3) / n_band
            # image_central[image_central > 0.7] = 1
            # image_central[image_central < 0.5] = 0
            #
            gkernel[image_central > 0.9] = 1
            image_central = image_central * gkernel
            image_central[image_central < 0.3] = 0
            image_central[image_central > 0.9] = 1
            # print(image_central.size())

            # aaa = image_central[0, :, :].cpu().detach().numpy()
            # prediction_data2 = pd.DataFrame(aaa)
            # data2 = pd.ExcelWriter('image_central.xlsx')  # 写入Excel文件
            # prediction_data2.to_excel(data2, 'image_central', float_format='%.5f')  # ‘page_1’是写入excel的sheet名
            # data2.save()
            # quit()
            # image_central = image_central.view(batch_size, -1)
            # print(image_central.size())
            # quit()

            # start = torch.cuda.Event(enable_timing=True)
            # end = torch.cuda.Event(enable_timing=True)
            #
            # start.record()

            coo_not_one_b, coo_not_one_x, coo_not_one_y = torch.where(image_central == 0.)
            coo_one_b, coo_one_x, coo_one_y = torch.where(image_central == 1.)

            num_not_one = np.array([(torch.where(coo_not_one_b == g)[0]).numel() for g in range(batch_size)])
            # print(num_not_one,num_not_one.shape, num_not_one.sum())
            num_one = np.array([(torch.where(coo_one_b == g)[0]).numel() for g in range(batch_size)])
            # print(num_one,num_one.shape, num_one.sum(),num_one[:2].sum())
            # quit()
            num_chazhi = np.where((num_not_one-num_one) > 0, num_one, num_not_one)
            # print(num_chazhi,num_chazhi.sum())

            # num_one_sum = [num_one[:a].sum() for a in range(batch_size)]
            # print(num_one_sum,len(num_one_sum))
            # num_not_one_sum = [num_not_one[:a].sum() for a in range(batch_size)]
            # print(num_not_one_sum, len(num_not_one_sum))
            # quit()

            coo = [(coo_not_one_b[a], coo_not_one_x[a], coo_not_one_y[a]) for a in range(coo_not_one_b.shape[0])]
            coo1 = [(coo_one_b[a], coo_one_x[a], coo_one_y[a]) for a in range(coo_one_b.shape[0])]
            # print(coo)
            # print(len(coo), coo[:2], coo[0][0], coo[0][0].item())
            # quit()
            coo2 = [coo1[num_one[:a].sum():(num_one[:a].sum()+num_chazhi[a])] for a in range(batch_size)]
            coo2 = list(itertools.chain(*coo2))
            coo2 = np.array(coo2)
            print(coo2)
            quit()

            coo3 = [coo[num_not_one[:a].sum():(num_not_one[:a].sum() + num_chazhi[a])] for a in range(batch_size)]
            coo3 = list(itertools.chain(*coo3))
            coo3 = np.array(coo3)

            # end.record()
            # # Waits for everything to finish running
            # torch.cuda.synchronize()
            # print(start.elapsed_time(end))

            # print(len(coo2), len(coo3))
            # #
            # print(coo2[30:40], '\n', coo3[30:40],coo2.shape,coo3.shape)
            # #
            # quit()

            for j in range(coo2.shape[0]):

                image[coo3[j][0].item(), :, coo3[j][1].item(), coo3[j][2].item()] = image[coo2[j][0].item(), :, coo2[j][1].item(), coo2[j][2].item()]

            #
            # num = np.array([(torch.where(coo_not_one_b == g)[0]).numel() for g in range(batch_size)])
            # print(num,num.shape,num.sum())
            # num = np.delete(num,np.where(num == 0))
            # print(num, num.shape, num.shape[0],num.sum())
            # # quit()
            # # print(coo_not_one_x[0], '\n')
            # coo1 = np.zeros((1, 3))
            # # print(coo1.shape, '\n')
            # # quit()
            # for b in range(num.shape[0]):
            #     print(b,num[b])
            #     coo = np.array(
            #         [(coo_not_one_b[a], coo_not_one_x[a], coo_not_one_y[a]) for a in range(num[b])])
            #     # print(coo.shape)
            #     # quit()
            #     coo1 = np.concatenate((coo1, coo), axis=0)
            #     # print(coo,coo.shape)
            #     # quit()
            # print('\n',coo1.shape)
            # quit()
            #
            # # coo1 = np.array([(coo_one_x[a], coo_one_y[a]) for a in range(coo_one_x.shape[0])])
            # #, coo.shape,coo[0],coo[0]-coo[0],coo[0][0])
            # # print(coo1.shape, coo1[0])
            #
            # quit()
            #
            #
            # for i in range(batch_size):
            #     # print(i)
            #     image_central_one = image_central[i, :]
            #     print(image_central_one.size())
            #     for j in range(patch_size*patch_size):
            #         coo_x = j // patch_size
            #         coo_y = j % patch_size
            #         print(coo_y)
            #         # quit()
            #         coo_xx = np.array([[x,y] for x in range(coo_x, coo_x + kernel_size) for y in range(coo_y, coo_y + kernel_size)])
            #         coo_xxx = np.array(
            #             [x * patch_size + y for x in range(coo_x, coo_x + kernel_size) for y in range(coo_y, coo_y + kernel_size)])
            #
            #         print(coo_xx,coo_xx.shape)
            #
            #         print(coo_xxx)
            #         aaa = torch.where(image_central_one[coo_xxx] == 0.)
            #
            #         quit()
            #
            #     for m in range(patch_size - kernel_size + 1):
            #         for n in range(patch_size - kernel_size + 1):
            #             # m,n = 10,5
            #             # print(m,n)
            #             # print(image_central_one[m:m + kernel_size, n:n + kernel_size])
            #
            #             coo_one_x,coo_one_y = torch.where(image_central_one[m:m + kernel_size, n:n + kernel_size] == 1.)
            #             real_coo_one_x, real_coo_one_y = m + coo_one_x,n + coo_one_y
            #             coo_not_one_x, coo_not_one_y = torch.where(image_central_one[m:m + kernel_size, n:n + kernel_size] != 1.)
            #             real_coo_not_one_x, real_coo_not_one_y = m + coo_not_one_x, n + coo_not_one_y
            #
            #             # print(coo_one_x,coo_one_y,real_coo_one_x, real_coo_one_y)
            #             # print(len(coo_one_x))
            #             # quit()
            #
            #             if len(coo_one_x) != kernel_size * kernel_size:
            #                 # print('len(coo_not_one_x) != kernel_size * kernel_size')
            #
            #                 def calculate_num_the_same_class(biger_size):
            #                     # print('biger_size',biger_size)
            #
            #                     x1, y1 = m,n
            #                     x2, y2 = x1 + kernel_size, y1 + kernel_size
            #
            #                     # print('x1,x2, y1,y2',x1,x2, y1,y2)
            #
            #                     x_begin = x1 - biger_size
            #                     x_end = x2 + biger_size
            #                     y_begin = y1 - biger_size
            #                     y_end = y2 + biger_size
            #                     if x_begin >= 0 and x_end <= patch_size and y_begin >= 0 and y_end <= patch_size:
            #                         pass
            #                     else:
            #                         if x_begin < 0:
            #                             x_begin = 0
            #                         if x_end > patch_size:
            #                             x_end = patch_size
            #                         if y_begin < 0:
            #                             y_begin = 0
            #                         if y_end > patch_size:
            #                             y_end = patch_size
            #
            #                     # print('x_begin:x_end, y_begin:y_end',x_begin,x_end, y_begin,y_end)
            #
            #                     image_central_one_out = image_central_one[x_begin:x_end, y_begin:y_end]
            #                     # print('输入 image_central_one_out 数据：', image_central_one_out)
            #
            #                     pad = nn.ZeroPad2d(padding=(y1 - y_begin, y_end - y2, x1 - x_begin, x_end - x2))
            #
            #                     yy = pad(image_central_one[m:m + kernel_size, n:n + kernel_size])
            #
            #                     image_central_one_out1 = image_central_one_out - yy
            #
            #                     coo_out_one_x, coo_out_one_y = torch.where(image_central_one_out1 == 1.)
            #
            #                     # print('\n', 'coo_out_one_x: ', '\n', coo_out_one_x, '\n', 'coo_out_one_y: ', '\n',coo_out_one_y)
            #                     # quit()
            #
            #                     if biger_size <= patch_size-1:
            #                         if len(coo_out_one_x) >= len(coo_not_one_x):
            #
            #                             real_coo_out_one_x = coo_out_one_x + x_begin
            #                             real_coo_out_one_y = coo_out_one_y + y_begin
            #
            #                             return real_coo_out_one_x, real_coo_out_one_y
            #                         else:
            #                             return calculate_num_the_same_class(biger_size + 1)
            #                     else:
            #                         real_coo_out_one_x = coo_out_one_x + x_begin
            #                         real_coo_out_one_y = coo_out_one_y + y_begin
            #                         return real_coo_out_one_x, real_coo_out_one_y
            #
            #                 real_coo_out_one_x1, real_coo_out_one_y1 = calculate_num_the_same_class(biger_size)
            #
            #                 # print('\n', 'real_coo_out_one_x: ', '\n', real_coo_out_one_x1, '\n', 'real_coo_out_one_y: ', '\n',real_coo_out_one_y1)
            #
            #                 for k in range(min(len(coo_not_one_x),len(real_coo_out_one_x1))):
            #                     img_one = img
            #                     # print(real_coo_not_one_x, real_coo_not_one_y)
            #                     # print(real_coo_not_one_x[k].item(),real_coo_not_one_y[k].item())
            #                     # print(real_coo_out_one_x1[k].item(),real_coo_out_one_y1[k].item())
            #                     img_one[i,real_coo_not_one_x[k].item(),real_coo_not_one_y[k].item(),:] = \
            #                         img[i,real_coo_out_one_x1[k].item(),real_coo_out_one_y1[k].item(),:]
            #
            #                     imgg1[i, m * kernel_size:m * kernel_size + kernel_size, n * kernel_size:n * kernel_size + kernel_size, :] = \
            #                         img_one[i, m:m + kernel_size, n:n + kernel_size, :]
            #             else:
            #                 # print('不需要可变卷积')
            #                 imgg1[i, m * kernel_size:m * kernel_size + kernel_size,n * kernel_size:n * kernel_size + kernel_size, :] = \
            #                     img[i, m:m + kernel_size, n:n + kernel_size, :]

            return image

            # image_central[image_central < 0.1] = 0
            # aa = image_central[0, :, :].cpu().detach().numpy()
            # prediction_data2 = pd.DataFrame(aa)
            # data2 = pd.ExcelWriter('image_central1.xlsx')  # 写入Excel文件
            # prediction_data2.to_excel(data2, 'image_central', float_format='%.5f')  # ‘page_1’是写入excel的sheet名
            # data2.save()
            # quit()

            # for i in range(batch_size):
            #     image_central_one = image_central[i, :, :]
            #     a = image_central_one.view(-1)
            #     aa, _ = torch.sort(a, descending=True) # 从大 到 小 排列
            #
            #     # image_central_one[image_central_one > aa[real_patchsize*real_patchsize - 1]] = 1
            #     image_central_one[image_central_one < aa[real_patchsize * real_patchsize - 1]] = 0
            #     image_central[i, :, :] = image_central_one

            # aa = image_central[0, :, :].cpu().detach().numpy()
            # prediction_data2 = pd.DataFrame(aa)
            # data2 = pd.ExcelWriter('image_central11.xlsx')  # 写入Excel文件
            # prediction_data2.to_excel(data2, 'image_central', float_format='%.5f')  # ‘page_1’是写入excel的sheet名
            # data2.save()
            # quit()



            # image_central = image_central.unsqueeze(3)
            # imgg = image_central.repeat(1, 1, 1, n_band)
            # imgg = imgg.permute(0, 3, 1, 2)
            # return imgg

        x_offset = bilateral_filter1(y, 10, 0.1).cuda()

        # x_offset = x_offset.permute(0, 3, 1, 2)
        out = self.conv(x_offset)
        # print('out size is : ', out.size())
        # quit()
        return out

# DeformConv2D_EPF6 是 只替换 0 和 1 的 数量最小值
class DeformConv2D_EPF6(nn.Module):

    def __init__(self, inc, outc, kernel_size=3, padding=1, stride=1):
        super(DeformConv2D_EPF6, self).__init__()

        self.conv = nn.Conv2d(inc, outc, kernel_size=3, stride=1, padding=1)

        # self.weight_1 = torch.nn.Parameter(torch.FloatTensor(1).cuda(), requires_grad=True)
        # self.weight_1.data.fill_(0.1)
        # self.weight_2 = torch.nn.Parameter(torch.FloatTensor(1).cuda(), requires_grad=True)
        # self.weight_2.data.fill_(10)

    def forward(self, x):

        y = x

        def bilateral_filter1(image, gsigma, ssigma):

            img = image.detach()
            batch_size = img.shape[0]
            n_band = img.shape[1]
            patch_size = img.shape[2]
            img = img.permute(0, 2, 3, 1)

            gkernel = gaus_kernel(patch_size, gsigma).cuda()
            gkernel = gkernel.repeat(batch_size, 1, 1)
            sigma2 = 2 * torch.mul(ssigma, ssigma).float()
            imgg = torch.zeros([batch_size, patch_size, patch_size, n_band])
            for i in range(batch_size):
                image_central = (img[i, :, :, :] - img[i, patch_size // 2, patch_size // 2, :]).cuda()
                imgg[i, :, :, :] = image_central

            image_central = torch.exp(-(imgg * imgg) / sigma2).cuda()
            image_central = torch.sum(image_central, 3) / n_band
            gkernel[image_central > 0.9] = 1
            image_central = image_central * gkernel
            image_central[image_central < 0.3] = 0
            image_central[image_central > 0.8] = 1
            image_central = image_central.cpu().numpy()

            coo_not_one_b, coo_not_one_x, coo_not_one_y = np.where(image_central == 0.)
            coo_one_b, coo_one_x, coo_one_y = np.where(image_central == 1.)
            # print(coo_not_one_b.shape, len(coo_not_one_b))

            if len(coo_not_one_b) > 0:
                # num_one_sum = torch.cumsum(coo_not_one_b,0)
                # coo = [(coo_not_one_b[a], coo_not_one_x[a], coo_not_one_y[a]) for a in range(coo_not_one_b.shape[0])]
                # coo1 = [(coo_one_b[a], coo_one_x[a], coo_one_y[a]) for a in range(coo_one_b.shape[0])]
                # print(coo[10:15])
                # print(len(coo), coo[:2], coo[0][0], coo[0][0].item())
                # quit()
                num1 = Counter(coo_not_one_b)
                num_not_one = np.array([num1[g] for g in range(batch_size)])
                # print(num1, '\n', num_not_one)
                # quit()

                num2 = Counter(coo_one_b)
                num_one = np.array([num2[g] for g in range(batch_size)])

                num_chazhi = np.where((num_not_one - num_one) > 0, num_one, num_not_one)
                # print(num_chazhi,num_chazhi.sum())

                coo_not_one_b = coo_not_one_b.reshape(len(coo_not_one_b), 1)
                coo_not_one_x = coo_not_one_x.reshape(len(coo_not_one_x), 1)
                coo_not_one_y = coo_not_one_y.reshape(len(coo_not_one_y), 1)
                coo_one_b = coo_one_b.reshape(len(coo_one_b), 1)
                coo_one_x = coo_one_x.reshape(len(coo_one_x), 1)
                coo_one_y = coo_one_y.reshape(len(coo_one_y), 1)

                coo333 = np.concatenate((coo_not_one_b, coo_not_one_x, coo_not_one_y), 1)
                coo444 = np.concatenate((coo_one_b, coo_one_x, coo_one_y), 1)
                # print(coo333.shape, coo444.shape)
                # quit()

                # print(num_chazhi,num_chazhi.sum())

                # num_one_sum = [num_one[:a].sum() for a in range(batch_size)]
                # print(num_one_sum,len(num_one_sum))
                # num_not_one_sum = [num_not_one[:a].sum() for a in range(batch_size)]
                # print(num_not_one_sum, len(num_not_one_sum))
                # quit()
                coo2 = [coo444[num_one[:a].sum():(num_one[:a].sum() + num_chazhi[a]), :] for a in range(batch_size)]
                coo2 = list(itertools.chain(*coo2))
                coo2 = np.array(coo2)
                # print(coo2.shape)

                coo3 = [coo333[num_not_one[:a].sum():(num_not_one[:a].sum() + num_chazhi[a]), :] for a in
                        range(batch_size)]
                coo3 = list(itertools.chain(*coo3))
                coo3 = np.array(coo3)
                # print(coo3.shape)

                coo3_b = coo3[:, 0]
                coo3_x = coo3[:, 1]
                coo3_y = coo3[:, 2]

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
                image[coo3_b, coo3_x, coo3_y] = image[coo2_b, coo2_x, coo2_y]
                image = image.permute(0, 3, 1, 2)
                return image
            else:
                return image

        x_offset = bilateral_filter1(y, 10, 0.1).cuda()

        # x_weighting[x_weighting > 0.9] = 1
        # x_weighting[x_weighting < self.weight_1] = 0

        # x_offset = x * x_weighting
        # x[x_weighting < 0.2] = 0
        # print(self.weight_1)
        # print(self.weight_2)
        # print(x_weighting.shape)
        # quit()
        # #
        # patch_size = x.shape[2]
        # real_patchsize = patch_size - 6
        # x_offset = x_offset.permute(0, 2, 3, 1)
        # x_offset_sum = torch.sum(x_offset, 3)
        # x_offset1 = x_offset[x_offset_sum != 0]
        # print(x_offset1.shape)
        # quit()
        # x_offset1 = x_offset1[0:x.shape[0]*real_patchsize*real_patchsize,:]
        # x_offset = x_offset1.view(x.shape[0],real_patchsize,real_patchsize,x.shape[1])
        # x_offset = x_offset.permute(0, 3, 1, 2)

        out = self.conv(x_offset)

        return out

# DeformConv2D_EPF7 是 全部替换不是同一类的
class DeformConv2D_EPF7(nn.Module):

    def __init__(self, inc, outc, kernel_size=3, padding=1, stride=1):
        super(DeformConv2D_EPF7, self).__init__()

        self.conv = nn.Conv2d(inc, outc, kernel_size=3, stride=1, padding=1)
        # self.weight_1 = torch.nn.Parameter(torch.FloatTensor(1).cuda(), requires_grad=True)
        # self.weight_1.data.fill_(0.1)
        # self.weight_2 = torch.nn.Parameter(torch.FloatTensor(1).cuda(), requires_grad=True)
        # self.weight_2.data.fill_(10)

    def forward(self, x):

        y = x

        def bilateral_filter1(image, gsigma, ssigma):

            img = image.detach()  # 截断反向传播
            batch_size = img.shape[0]
            n_band = img.shape[1]
            patch_size = img.shape[2]
            img = img.permute(0, 2, 3, 1)

            gkernel = gaus_kernel(patch_size, gsigma).cuda()  # 生成 3*3 空间域权重
            gkernel = gkernel.repeat(batch_size, 1, 1)   # 个数扩展到 batch_size 个
            sigma2 = 2 * torch.mul(ssigma, ssigma).float()
            imgg = torch.zeros([batch_size, patch_size, patch_size, n_band])
            # 每个 batch_size 减去各自的中心像素的光谱曲线
            for i in range(batch_size):
                image_central = (img[i, :, :, :] - img[i, patch_size // 2, patch_size // 2, :]).cuda()
                imgg[i, :, :, :] = image_central

            # 将得到的差值，送入高斯核，计算相似度矩阵
            # sigma2 = 0.002
            image_central = torch.exp(-(imgg * imgg) / sigma2).cuda()
            image_central = torch.sum(image_central, 3) / n_band
            gkernel[image_central > 0.9] = 1
            # 得到 相似度 矩阵， 空间域和光谱域相乘
            image_central = image_central * gkernel
            # print(image_central.shape)
            # quit()
            # 将 相似度矩阵 离散化， 设置 两个阈值，高于0.9认为是和中心像素是同一类样本，低于0.2认为和中心像素不同类样本

            # image_central[image_central < 0.2] = 0  # for paviau
            # image_central[image_central > 0.9] = 1  # for paviau

            # image_central[image_central < 0.5] = 0  # for SD
            # image_central[image_central > 0.9] = 1  # for SD

            image_central[image_central < 0.2] = 0  # for houston2018
            image_central[image_central > 0.9] = 1  # for houston2018

            image_central = image_central.cpu().numpy()
            # 在excel中，查看生成的相似度矩阵
            # aaa = image_central[0, :, :]
            # prediction_data2 = pd.DataFrame(aaa)
            # data2 = pd.ExcelWriter('image_central.xlsx')  # 写入Excel文件
            # prediction_data2.to_excel(data2, 'image_central', float_format='%.5f')  # ‘page_1’是写入excel的sheet名
            # data2.save()
            # quit()

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
            # print(coo_not_one_b.shape, len(coo_not_one_b))

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
                image[coo_not_one_b, coo_not_one_x, coo_not_one_y] = image[coo2_b, coo2_x, coo2_y]
                image = image.permute(0, 3, 1, 2)
                return image
            else:
                return image

        x_offset = bilateral_filter1(y, 10, 0.1).cuda()  # for paviau
        #
        # x_offset = bilateral_filter1(y, 10, 0.02).cuda()  # for sd
        #
        # x_offset = bilateral_filter1(y, 10, 0.04).cuda()  # for houston2018

        # x_weighting[x_weighting > 0.9] = 1
        # x_weighting[x_weighting < self.weight_1] = 0

        # x_offset = x * x_weighting
        # x[x_weighting < 0.2] = 0
        # print(self.weight_1)
        # print(self.weight_2)
        # print(x_weighting.shape)
        # quit()
        # #
        # patch_size = x.shape[2]
        # real_patchsize = patch_size - 6
        # x_offset = x_offset.permute(0, 2, 3, 1)
        # x_offset_sum = torch.sum(x_offset, 3)
        # x_offset1 = x_offset[x_offset_sum != 0]
        # print(x_offset1.shape)
        # quit()
        # x_offset1 = x_offset1[0:x.shape[0]*real_patchsize*real_patchsize,:]
        # x_offset = x_offset1.view(x.shape[0],real_patchsize,real_patchsize,x.shape[1])
        # x_offset = x_offset.permute(0, 3, 1, 2)
        out = self.conv(x_offset)  # 正常卷积

        return out

# DeformConv2D_EPF8 是 全部替换不是同一类的
# 利用  双边滤波  CV 中函数  生成 相似度
class DeformConv2D_EPF8(nn.Module):

    def __init__(self, inc, outc, kernel_size=3, padding=1, stride=1):
        super(DeformConv2D_EPF8, self).__init__()

        self.conv = nn.Conv2d(inc, outc, kernel_size=3, stride=1, padding=1)

        # self.weight_1 = torch.nn.Parameter(torch.FloatTensor(1).cuda(), requires_grad=True)
        # self.weight_1.data.fill_(0.1)
        # self.weight_2 = torch.nn.Parameter(torch.FloatTensor(1).cuda(), requires_grad=True)
        # self.weight_2.data.fill_(10)

    def forward(self, x):

        y = x
        # print(y.shape)

        def bilateral_filter1(image, gsigma, ssigma):

            img = image[:, 103:, :, :].detach()
            batch_size = img.shape[0]
            n_band = img.shape[1]
            patch_size = img.shape[2]
            img = img.permute(0, 2, 3, 1)

            # gkernel = gaus_kernel(patch_size, gsigma).cuda()
            # gkernel = gkernel.repeat(batch_size, 1, 1)
            sigma2 = 2 * torch.mul(ssigma, ssigma).float()
            imgg = torch.zeros([batch_size, patch_size, patch_size, n_band])
            for i in range(batch_size):
                image_central = (img[i, :, :, :] - img[i, patch_size // 2, patch_size // 2, :]).cuda()
                imgg[i, :, :, :] = image_central

            image_central = torch.exp(-(imgg * imgg) / sigma2).cuda()
            image_central = torch.sum(image_central, 3) / n_band
            # aaa = image_central[0, :, :].cpu().numpy()
            # prediction_data2 = pd.DataFrame(aaa)
            # data2 = pd.ExcelWriter('image_central.xlsx')  # 写入Excel文件
            # prediction_data2.to_excel(data2, 'image_central', float_format='%.5f')  # ‘page_1’是写入excel的sheet名
            # data2.save()
            # quit()
            image_central[image_central < 0.2] = 0
            image_central[image_central > 0.9] = 1
            image_central = image_central.cpu().numpy()
            # aaa = image_central[0, :, :]
            # prediction_data2 = pd.DataFrame(aaa)
            # data2 = pd.ExcelWriter('image_central.xlsx')  # 写入Excel文件
            # prediction_data2.to_excel(data2, 'image_central', float_format='%.5f')  # ‘page_1’是写入excel的sheet名
            # data2.save()
            # quit()
            coo_not_one_b, coo_not_one_x, coo_not_one_y = np.where(image_central == 0.)
            coo_one_b, coo_one_x, coo_one_y = np.where(image_central == 1.)
            # print(coo_not_one_b.shape, len(coo_not_one_b))

            if len(coo_not_one_b) > 0:
                # num_one_sum = torch.cumsum(coo_not_one_b,0)
                # coo = [(coo_not_one_b[a], coo_not_one_x[a], coo_not_one_y[a]) for a in range(coo_not_one_b.shape[0])]
                # coo1 = [(coo_one_b[a], coo_one_x[a], coo_one_y[a]) for a in range(coo_one_b.shape[0])]
                # print(coo[10:15])
                # print(len(coo), coo[:2], coo[0][0], coo[0][0].item())
                # quit()
                num1 = Counter(coo_not_one_b)
                num_not_one = np.array([num1[g] for g in range(batch_size)])
                # print('num_not_one','\n', num_not_one,num_not_one.sum())
                # quit()

                num2 = Counter(coo_one_b)
                num_one = np.array([num2[g] for g in range(batch_size)])
                # print('num_one','\n', num_one,num_one.sum())

                num_chazhi = np.where((num_not_one - num_one) > 0, num_one, num_not_one)
                # print('num_chazhi','\n',num_chazhi, num_chazhi.sum())
                # quit()

                coo_not_one_b = coo_not_one_b.reshape(len(coo_not_one_b), 1)
                coo_not_one_x = coo_not_one_x.reshape(len(coo_not_one_x), 1)
                coo_not_one_y = coo_not_one_y.reshape(len(coo_not_one_y), 1)
                coo_one_b = coo_one_b.reshape(len(coo_one_b), 1)
                coo_one_x = coo_one_x.reshape(len(coo_one_x), 1)
                coo_one_y = coo_one_y.reshape(len(coo_one_y), 1)

                coo333 = np.concatenate((coo_not_one_b, coo_not_one_x, coo_not_one_y), 1)
                coo444 = np.concatenate((coo_one_b, coo_one_x, coo_one_y), 1)
                # print(coo333.shape, coo444.shape)
                # quit()

                # num_one_sum = [num_one[:a].sum() for a in range(batch_size)]
                # print('num_one_sum','\n',num_one_sum,len(num_one_sum))
                # num_not_one_sum = [num_not_one[:a].sum() for a in range(batch_size)]
                # print('num_not_one_sum','\n',num_not_one_sum, len(num_not_one_sum))
                # quit()

                coo2 = [coo444[num_one[:a].sum():(num_one[:a].sum() + num_chazhi[a]), :] for a in range(batch_size)]
                coo2 = list(itertools.chain(*coo2))
                coo2 = np.array(coo2)
                # print(coo2.shape)

                num_chazhi_zeros = np.where((num_not_one - num_one) > 0)[0]
                # print('num_chazhi_zeros','\n',num_chazhi_zeros, len(num_chazhi_zeros))
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

                coo3_b = coo333[:, 0]
                coo3_x = coo333[:, 1]
                coo3_y = coo333[:, 2]

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
                image[coo3_b, coo3_x, coo3_y] = image[coo2_b, coo2_x, coo2_y]

                image = image.permute(0, 3, 1, 2)
                return image
            else:
                return image

        x_offset = bilateral_filter1(y, 10, 1).cuda()
        x_offset = x_offset[:, :103, :, :]
        # print(x_offset.shape)
        # quit()


        # x_weighting[x_weighting > 0.9] = 1
        # x_weighting[x_weighting < self.weight_1] = 0

        # x_offset = x * x_weighting
        # x[x_weighting < 0.2] = 0
        # print(self.weight_1)
        # print(self.weight_2)
        # print(x_weighting.shape)
        # quit()
        # #
        # patch_size = x.shape[2]
        # real_patchsize = patch_size - 6
        # x_offset = x_offset.permute(0, 2, 3, 1)
        # x_offset_sum = torch.sum(x_offset, 3)
        # x_offset1 = x_offset[x_offset_sum != 0]
        # print(x_offset1.shape)
        # quit()
        # x_offset1 = x_offset1[0:x.shape[0]*real_patchsize*real_patchsize,:]
        # x_offset = x_offset1.view(x.shape[0],real_patchsize,real_patchsize,x.shape[1])
        # x_offset = x_offset.permute(0, 3, 1, 2)

        out = self.conv(x_offset)

        return out