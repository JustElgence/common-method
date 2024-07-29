import torch.nn as nn
import numpy as np
import torch
from datasets import get_dataset
from utils import display_dataset,bilateralFilter_img
import visdom
import pandas as pd
from network.deform_conv import deform_conv, deform_conv_v2,deform_conv_epf
# Load the dataset  此时输出的 img 是归一化之后的 img


a = torch.torch.rand(2, 3)

# print(a.size(),a)
a = np.reshape(a,(a.shape[0], a.shape[1], 1))
print(a.size(),a)
a1 = a
while a1.shape[2] < 4:

    a1 = np.concatenate([a1, a], 2)
    print(a1.shape)
print(a1)
a1 = a1.transpose((2, 0, 1))
print(a1)
quit()








device='cuda:0'
DATASET = 'PaviaU'
MODEL = 'cnn'
viz = visdom.Visdom(env=DATASET + ' ' + MODEL)

img, gt, LABEL_VALUES, IGNORED_LABELS, RGB_BANDS, palette = get_dataset('PaviaU',"./Datasets/")

img_bilateralFilter_sum = bilateralFilter_img(img, 3, 20, 20)  # 先 使用 这种试试结果

# prediction_data2 = pd.DataFrame(img_bilateralFilter_sum)
# data2 = pd.ExcelWriter('img_bilateralFilter_sum.xlsx')  # 写入Excel文件
# prediction_data2.to_excel(data2, 'img_bilateralFilter_sum', float_format='%.5f')  # ‘page_1’是写入excel的sheet名
# data2.save()

class CNN(nn.Module):
    def __init__(self,input_channels, n_classes):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(
            deform_conv_epf.DeformConv2D_EPF1(input_channels, 32, kernel_size=3, stride=1, padding=1, ),
            # nn.Conv2d(in_channels=input_channels,out_channels=32,kernel_size=3,stride=1,padding=1,),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            # nn.MaxPool2d(kernel_size=2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            # nn.MaxPool2d(2),
            # nn.Dropout(0.5),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            # nn.AdaptiveMaxPool2d(1),
            #nn.Dropout(0.5),
        )
        self.out = nn.Linear(6272, 3136)  # fully connected layer, output 16 classes
        self.out1 = nn.Linear(3136, n_classes)  # fully connected layer, output 16 classes

    def forward(self, x):
        print(x.size())
        x = self.conv1(x)
        print(x.size())
        # x = self.conv2(x)
        # print(x.size())
        # x = self.conv3(x)
        # print(x.size())
        # x = x.view(x.size(0), -1)  # flatten the output of conv2 to (batch_size, 32 * 7 * 7)
        # print(x.size())
        # output = self.out(x)
        # print(output.size())
        # output = self.out1(output)
        # print(output.size())

        return x

model = CNN(img.shape[2], 10)

img = np.asarray(np.copy(img).transpose((2, 0, 1)), dtype='float32')
img = torch.from_numpy(img)
img = torch.unsqueeze(img,0).to(device)
model.to(device)


outs = model(img)


outs = torch.squeeze(outs,0)
outs = outs.cpu().detach().numpy()
outs = outs.transpose((1, 2, 0))

outs_bilateralFilter_sum = bilateralFilter_img(outs, 3, 20, 20)  # 先 使用 这种试试结果
prediction_data2 = pd.DataFrame(outs_bilateralFilter_sum)
data2 = pd.ExcelWriter('outs_bilateralFilter_sum1.xlsx')  # 写入Excel文件
prediction_data2.to_excel(data2, 'outs_bilateralFilter_sum', float_format='%.5f')  # ‘page_1’是写入excel的sheet名
data2.save()

display_dataset(outs, gt, (0, 1, 3), LABEL_VALUES, palette, viz)

