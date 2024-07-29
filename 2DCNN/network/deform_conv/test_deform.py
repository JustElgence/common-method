# -*- coding: utf-8 -*-
"""
 @Time    : 2019/2/20 22:16
 @Author  : Wang Xin
 @Email   : wangxin_buaa@163.com
"""

import numpy as np

import torch
import torch.nn as nn


class DeformConv2D(nn.Module):

    def __init__(self, inc, outc, kernel_size=3, padding=1, stride=1, bias=None,
                 lr_ratio=1.0):  # input is : 64, 128, 3, 1, 1
        super(DeformConv2D, self).__init__()
        self.kernel_size = kernel_size
        # print(kernel_size)
        # quit()
        self.padding = padding
        self.stride = stride
        self.zero_padding = nn.ZeroPad2d(padding)  # ZeroPad2d : 使用0填充输入tensor的边界

        self.offset_conv = nn.Conv2d(inc, 2 * kernel_size * kernel_size, kernel_size=3, padding=1, stride=stride)
        nn.init.constant_(self.offset_conv.weight, 0)  # the offset learning are initialized with zero weights
        self.offset_conv.register_backward_hook(self._set_lr)

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
        print('run deform_conv, the input data size', x.size())
        offset = self.offset_conv(x)
        print('run deform_conv, the offset size', offset.size())
        dtype = offset.data.type()
        print('offset.data.type() : ', dtype)
        ks = self.kernel_size
        print('ks = self.kernel_size,ks is : ', ks)
        N = offset.size(1) // 2
        print('run deform_conv, N = offset.size(1) // 2, the N size : ', N)
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
            print('对输入的 X 进行 0 填充')
            x = self.zero_padding(x)
            print('run deform_conv, 对输入的 X 进行 0 填充后，the input data X  size', x.size())
            print('***************************分隔符**************************')

        # (b, 2N, h, w)
        p = self._get_p(offset, dtype)
        print('self._get_p, p size is : ', p.size())

        # (b, h, w, 2N)  就是 维度 转换
        p = p.contiguous().permute(0, 2, 3, 1)
        print('p.contiguous().permute', p.size(), '\n', 'p is : ', '\n', p)
        # quit()
        """
            if q is float, using bilinear interpolate, it has four integer corresponding position.
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
        q_lt = p.detach().floor()  # 不更新 参数  ，， 取整
        print('q_lt = p.detach().floor(), q_lt : ', q_lt.size(), '\n', 'q_lt is : ', q_lt)
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
        print('q_rb is : ', q_rb.size(), q_rb)

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
        q_lt = torch.cat([torch.clamp(q_lt[..., :N], 0, x.size(2) - 1), torch.clamp(q_lt[..., N:], 0, x.size(3) - 1)],
                         dim=-1).long()
        print('q_lt = torch.cat , q_lt : ', q_lt.size(), q_lt)
        # quit()

        """
            x.size(2) is h, x.size(3) is w, make 0 <= p_y <= h - 1, 0 <= p_x <= w-1
        """
        q_rb = torch.cat([torch.clamp(q_rb[..., :N], 0, x.size(2) - 1), torch.clamp(q_rb[..., N:], 0, x.size(3) - 1)],
                         dim=-1).long()
        print('q_rb = torch.cat , q_rb : ', q_rb.size())

        """
            For the left bottom point, it'x is equal to right bottom, it'y is equal to left top
            Therefore, it's y is from q_lt, it's x is from q_rb
        """
        q_lb = torch.cat([q_lt[..., :N], q_rb[..., N:]], -1)
        print('q_lb = torch.cat , q_lb : ', q_lb.size())
        """
            y from q_rb, x from q_lt
            For right top point, it's x is equal t to left top, it's y is equal to right bottom 
        """
        q_rt = torch.cat([q_rb[..., :N], q_lt[..., N:]], -1)
        print('q_rt = torch.cat , q_rt : ', q_rt.size())
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

        print('g_lt size is ', g_lt.size())
        print('g_rb size is ', g_rb.size())
        print('g_lb size is ', g_lb.size())
        print('g_rt size is ', g_rt.size())
        quit()
        # print('g_lt unsqueeze size:', g_lt.unsqueeze(dim=1).size())

        # (b, c, h, w, N)
        x_q_lt = self._get_x_q(x, q_lt, N)
        x_q_rb = self._get_x_q(x, q_rb, N)
        x_q_lb = self._get_x_q(x, q_lb, N)
        x_q_rt = self._get_x_q(x, q_rt, N)

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

        out = self.conv(x_offset)
        print('out size is : ', out.size())
        return out

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
        print('p_n_x is : ', '\n', p_n_x, '\n', ' p_n_y is : ', '\n', p_n_y)
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
        print('p_n.size() is : ', p_n.size(), '\n', 'p_n is : ', '\n', p_n)
        '''
        p_n.size() is :  torch.Size([18])
        tensor([-1, -1, -1,  0,  0,  0,  1,  1,  1, -1,  0,  1, -1,  0,  1, -1,  0,  1])
        '''
        p_n = p_n.view(1, 2 * N, 1, 1).type(dtype)
        p_n.requires_grad = False
        print('requires_grad:', p_n.requires_grad)
        print('p_n.size() is : ', p_n.size(), '\n', p_n)
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
        print('p_0 is ；', p_0, '\n', 'p_0 size is : ', p_0.size())
        # quit()
        p_0.requires_grad = False

        return p_0

    def _get_p(self, offset, dtype):
        N, h, w = offset.size(1) // 2, offset.size(2), offset.size(3)
        print('***************************分隔符**************************')
        print('offset.size(1) // 2, offset.size(2), offset.size(3), N,h,w is : ', N, h, w)

        # (1, 2N, 1, 1)
        p_n = self._get_p_n(N, dtype)  # p_n 是 一个确定的矩阵
        print('p_n is : ', p_n.size())
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
        print('p_0 is : ', p_0.size())
        print('***************************分隔符**************************')
        # quit()

        print('p_0 size is : ', p_0.size())
        print('p_n size is : ', p_n.size())
        print('offset size is : ', offset.size())
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
        print('p_test_no_use is : ', '\n', p_test_no_use)
        p = p_0 + p_n + offset
        print('p is :', '\n', p)
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

        x_offset = x.gather(dim=-1, index=index).contiguous().view(b, c, h, w, N)

        return x_offset

    @staticmethod
    def _reshape_x_offset(x_offset, ks):
        b, c, h, w, N = x_offset.size()
        x_offset = torch.cat([x_offset[..., s:s + ks].contiguous().view(b, c, h, w * ks) for s in range(0, N, ks)],
                             dim=-1)
        x_offset = x_offset.contiguous().view(b, c, h * ks, w * ks)

        return x_offset


from network.deform_conv.deform_conv import DeformConv2D as DeformConv2D_ori
from time import time

if __name__ == '__main__':
    # x = torch.randn(4, 3, 255, 255)

    # p_conv = nn.Conv2d(3, 2 * 3 * 3, kernel_size=3, padding=1, stride=1)
    # conv = nn.Conv2d(3, 64, kernel_size=3, stride=3, bias=False)
    #
    # d_conv1 = DeformConv2D(3, 64)
    # d_conv2 = DeformConv2D_ori(3, 64)
    #
    # offset = p_conv(x)
    #
    # end = time()
    # y1 = conv(d_conv1(x, offset))
    # end = time() - end
    # print('#1 speed = ', end)
    #
    # end = time()
    # y2 = conv(d_conv2(x, offset))
    # end = time() - end
    # print('#2 speed = ', end)

    # mask = (y1 == y2)
    # print(mask)
    # print(torch.max(mask))
    # print(torch.min(mask))

    x = torch.randn(4, 3, 255, 255)
    d_conv = DeformConv2D(3, 64)
    conv = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)

    end = time()
    y = d_conv(x)
    end = time() - end
    print('speed = ', end)
    print(y.size())

    end = time()
    y = conv(x)
    end = time() - end
    print('speed = ', end)

    if isinstance(d_conv, nn.Conv2d):
        print('Yes')
    else:
        print('No')




