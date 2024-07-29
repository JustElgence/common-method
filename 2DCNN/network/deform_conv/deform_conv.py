# -*- coding: utf-8 -*-
"""
 @Time    : 2019/2/20 22:16
 @Author  : Wang Xin
 @Email   : wangxin_buaa@163.com
"""

import numpy as np
import torch
import torch.nn as nn

class DeformConv2D1(nn.Module):

    def __init__(self, inc, outc, kernel_size=3, padding=1, stride=1, bias=None, lr_ratio=1.0): #  input is : 64, 128, 3, 1, 1
        super(DeformConv2D1, self).__init__()
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
        # print('run deform_conv, the input data size',x.size())
        offset = self.offset_conv(x)
        # print('run deform_conv, the offset size', offset.size())
        # print(offset[0,:,0,0])
        dtype = offset.data.type()
        # print('offset.data.type() : ',dtype)
        ks = self.kernel_size
        # print('ks = self.kernel_size,ks is : ',ks)
        N = offset.size(1) // 2
        # print('run deform_conv, N = offset.size(1) // 2, the N size : ', N)
        #quit()

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
        # print('self._get_p, p size is : ',p.size())

        # (b, h, w, 2N)  就是 维度 转换
        p = p.contiguous().permute(0, 2, 3, 1)
        #print('p.contiguous().permute',p.size(),'\n','p is : ','\n',p)    # 《Deformable Convolutional Networks》中 公式1 或者公式2 中的 p
        #quit()
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
        q_lt = p.detach().floor()   #  不更新 参数  ，， 向下取整，就是舍弃小数
        # print('q_lt = p.detach().floor(), q_lt : ',q_lt.size(),'\n','q_lt is : ','\n',q_lt,'\n','q_lt[..., :N] is : ','\n',q_lt[..., :N])
        # print(q_lt[0,1,1,:])
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
        #print('q_rb is : ',q_rb.size(),q_rb)

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
        #目标坐标需要在图片最大坐标范围内，将目标坐标进行切割限制
        q_lt = torch.cat([torch.clamp(q_lt[..., :N], 0, x.size(2) - 1), torch.clamp(q_lt[..., N:], 0, x.size(3) - 1)],
                         dim=-1).long()
        #print('q_lt = torch.cat , q_lt : ', q_lt.size(),q_lt)
        #quit()

        """
            x.size(2) is h, x.size(3) is w, make 0 <= p_y <= h - 1, 0 <= p_x <= w-1
        """
        q_rb = torch.cat([torch.clamp(q_rb[..., :N], 0, x.size(2) - 1), torch.clamp(q_rb[..., N:], 0, x.size(3) - 1)],
                         dim=-1).long()
        #print('q_rb = torch.cat , q_rb : ', q_rb.size())

        """
            For the left bottom point, it'x is equal to right bottom, it'y is equal to left top
            Therefore, it's y is from q_lt, it's x is from q_rb
        """
        q_lb = torch.cat([q_lt[..., :N], q_rb[..., N:]], -1)
        #print('q_lb = torch.cat , q_lb : ', q_lb.size())
        """
            y from q_rb, x from q_lt
            For right top point, it's x is equal t to left top, it's y is equal to right bottom 
        """
        q_rt = torch.cat([q_rb[..., :N], q_lt[..., N:]], -1)
        #print('q_rt = torch.cat , q_rt : ', q_rt.size())
        #quit()
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
        # #quit()
        # print('g_lt unsqueeze size:', g_lt.unsqueeze(dim=1).size())

        # (b, c, h, w, N)
        # print(' the input x.size() : ',x.size())
        x_q_lt = self._get_x_q(x, q_lt, N)
        x_q_rb = self._get_x_q(x, q_rb, N)
        x_q_lb = self._get_x_q(x, q_lb, N)
        x_q_rt = self._get_x_q(x, q_rt, N)
        # print('x_q_lt size is ', x_q_lt.size())
        # quit()
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
        # print('x_offset is : ', x_offset.size())
        x_offset = self._reshape_x_offset(x_offset, ks)
        # print('x_offset is : ',x_offset.size())
        #2.Get all integrated pixels into  a new image as input of  next layer.The rest is same as CNN
        # quit()

        out = self.conv(x_offset)
        # print('out size is : ',out.size())
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
             torch.arange(-(self.kernel_size - 1) // 2, (self.kernel_size - 1) // 2 + 1)])      # kernel_size is 3
                                                                                                # 就是 等价于 torch.arange(-1,2)
        #print('p_n_x is : ','\n',p_n_x,'\n',' p_n_y is : ','\n',p_n_y)
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
        #print('p_n.size() is : ',p_n.size(),'\n','p_n is : ','\n',p_n)
        '''
        p_n.size() is :  torch.Size([18])
        tensor([-1, -1, -1,  0,  0,  0,  1,  1,  1, -1,  0,  1, -1,  0,  1, -1,  0,  1])
        '''
        p_n = p_n.view(1, 2 * N, 1, 1).type(dtype)
        p_n.requires_grad = False
        #print('requires_grad:', p_n.requires_grad)
        #print('p_n.size() is : ',p_n.size(),'\n',p_n)
        '''
        p_n.size() is :  torch.Size([1, 18, 1, 1])
        '''
        #quit()
        return p_n
    def _get_p_0(self, h, w, N, dtype):
        p_0_x, p_0_y = torch.meshgrid([
            torch.arange(1, h * self.stride + 1, self.stride),
            torch.arange(1, w * self.stride + 1, self.stride)])
        p_0_x = torch.flatten(p_0_x).view(1, 1, h, w).repeat(1, N, 1, 1)
        p_0_y = torch.flatten(p_0_y).view(1, 1, h, w).repeat(1, N, 1, 1)
        p_0 = torch.cat([p_0_x, p_0_y], 1).type(dtype)
        #print('p_0 is ；',p_0,'\n','p_0 size is : ',p_0.size())
        #quit()
        p_0.requires_grad = False

        return p_0
####################################################################

    def _get_p(self, offset, dtype):
        N, h, w = offset.size(1) // 2, offset.size(2), offset.size(3)
        # print('***************************分隔符**************************')
        # print('offset.size(1) // 2, offset.size(2), offset.size(3), N,h,w is : ',N,h,w)

        # (1, 2N, 1, 1)
        p_n = self._get_p_n(N, dtype)   # p_n 是 一个确定的矩阵
        #  求的就是  《Deformable Convolutional Networks》中 公式1 或者公式2 中的 pn
        #  pn enumerates the locations in R.  R = {(−1,−1), (−1, 0), . . . , (0, 1), (1, 1)}
        #  - - pn_x - -     - - pn_y - -
        #   -1 -1  -1        -1  0   1
        #    0  0   0        -1  0   1           中间的就是 中心像素 p_0
        #    1  1   1        -1  0   1
        #  - - - - - -      - - - - - -
        #print('p_n is : ',p_n.size())
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
        #quit()

        # print('p_0 size is : ', p_0.size())
        # print('p_n size is : ', p_n.size())
        # print('offset size is : ', offset.size())
        #quit()
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
        # print(q.size())
        # print(x.size())
        x = x.contiguous().view(b, c, -1)
        # print(x.size())

        # (b, h, w, N)
        index = q[..., :N] * padded_w + q[..., N:]  # offset_x*w + offset_y
        # (b, c, h*w*N)
        index = index.contiguous().unsqueeze(dim=1).expand(-1, c, -1, -1, -1).contiguous().view(b, c, -1)
        # print(index.size())
        # quit()
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
        # print(ks)
        x_offset = torch.cat([x_offset[..., s:s + ks].contiguous().view(b, c, h, w * ks) for s in range(0, N, ks)],
                             dim=-1)
        # print('x_offset is : ', x_offset.size())
        x_offset = x_offset.contiguous().view(b, c, h * ks, w * ks)

        return x_offset
class DeformConv2D(nn.Module):
    def __init__(self, inc, outc, kernel_size=3, padding=1, stride=1, bias=None, lr_ratio=1.0):
        super(DeformConv2D, self).__init__()
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.zero_padding = nn.ZeroPad2d(padding)

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
        offset = self.offset_conv(x)
        dtype = offset.data.type()
        ks = self.kernel_size
        N = offset.size(1) // 2

        # Change offset's order from [x1, x2, ..., y1, y2, ...] to [x1, y1, x2, y2, ...]
        # Codes below are written to make sure same results of MXNet implementation.
        # You can remove them, and it won't influence the module's performance.
        offsets_index = torch.cat([torch.arange(0, 2 * N, 2), torch.arange(1, 2 * N + 1, 2)]).type_as(x).long()
        offsets_index.requires_grad = False
        offsets_index = offsets_index.unsqueeze(dim=0).unsqueeze(dim=-1).unsqueeze(dim=-1).expand(*offset.size())
        offset = torch.gather(offset, dim=1, index=offsets_index)
        # ------------------------------------------------------------------------

        if self.padding:
            x = self.zero_padding(x)

        # (b, 2N, h, w)
        p = self._get_p(offset, dtype)

        # (b, h, w, 2N)
        p = p.contiguous().permute(0, 2, 3, 1)

        """
            if q is float, using bilinear interpolate, it has four integer corresponding position.
            The four position is left top, right top, left bottom, right bottom, defined as q_lt, q_rb, q_lb, q_rt
        """
        # (b, h, w, 2N)
        q_lt = p.detach().floor()

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

        """
            x.size(2) is h, x.size(3) is w, make 0 <= p_y <= h - 1, 0 <= p_x <= w-1
        """
        q_lt = torch.cat([torch.clamp(q_lt[..., :N], 0, x.size(2) - 1), torch.clamp(q_lt[..., N:], 0, x.size(3) - 1)],
                         dim=-1).long()

        """
            x.size(2) is h, x.size(3) is w, make 0 <= p_y <= h - 1, 0 <= p_x <= w-1
        """
        q_rb = torch.cat([torch.clamp(q_rb[..., :N], 0, x.size(2) - 1), torch.clamp(q_rb[..., N:], 0, x.size(3) - 1)],
                         dim=-1).long()

        """
            For the left bottom point, it'x is equal to right bottom, it'y is equal to left top
            Therefore, it's y is from q_lt, it's x is from q_rb
        """
        q_lb = torch.cat([q_lt[..., :N], q_rb[..., N:]], -1)

        """
            y from q_rb, x from q_lt
            For right top point, it's x is equal t to left top, it's y is equal to right bottom 
        """
        q_rt = torch.cat([q_rb[..., :N], q_lt[..., N:]], -1)

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
           When thr point in the padding area, interpolation is not meaningful and we can take the nearest
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

        # print('g_lt size is ', g_lt.size())
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
        return out

    def _get_p_n(self, N, dtype):
        """
            In torch 0.4.1 grid_x, grid_y = torch.meshgrid([x, y])
            In torch 1.0   grid_x, grid_y = torch.meshgrid(x, y)
        """
        p_n_x, p_n_y = torch.meshgrid(
            [torch.arange(-(self.kernel_size - 1) // 2, (self.kernel_size - 1) // 2 + 1),
             torch.arange(-(self.kernel_size - 1) // 2, (self.kernel_size - 1) // 2 + 1)])
        # (2N, 1)
        p_n = torch.cat([torch.flatten(p_n_x), torch.flatten(p_n_y)], 0)
        p_n = p_n.view(1, 2 * N, 1, 1).type(dtype)
        p_n.requires_grad = False
        # print('requires_grad:', p_n.requires_grad)

        return p_n

    def _get_p_0(self, h, w, N, dtype):
        p_0_x, p_0_y = torch.meshgrid([
            torch.arange(1, h * self.stride + 1, self.stride),
            torch.arange(1, w * self.stride + 1, self.stride)])
        p_0_x = torch.flatten(p_0_x).view(1, 1, h, w).repeat(1, N, 1, 1)
        p_0_y = torch.flatten(p_0_y).view(1, 1, h, w).repeat(1, N, 1, 1)
        p_0 = torch.cat([p_0_x, p_0_y], 1).type(dtype)
        p_0.requires_grad = False

        return p_0

    def _get_p(self, offset, dtype):
        N, h, w = offset.size(1) // 2, offset.size(2), offset.size(3)

        # (1, 2N, 1, 1)
        p_n = self._get_p_n(N, dtype)

        # (1, 2N, h, w)
        p_0 = self._get_p_0(h, w, N, dtype)

        p = p_0 + p_n + offset
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