# -*- coding: utf-8 -*-
"""
2020
@author: wenxiang zhu
email: zhuwenxiang@hrbeu.edu.cn
"""

import os
import scipy.io as sio
from spectral import *

dataset_path = os.path.join('Datasets\PaviaU') # 数据集路径
data = sio.loadmat(os.path.join(dataset_path, 'PaviaU.mat'))['paviaU']

data = data[302:334, 93:129, :]
print(data.shape)
name = 'one.mat'
sio.savemat(name, {'one': data})
# spectral.settings.WX_GL_DEPTH_SIZE = 100
#
#
# view_cube(data, bands=[55, 41, 12])
#