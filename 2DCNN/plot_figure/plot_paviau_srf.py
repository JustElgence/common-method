
import matplotlib as mpl
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import ticker

x = ["1683",   "2276", "2597",  "3062",  "3506",  "3934",  "4201",  "4391",  "4474", "4768"]
y = [98.09, 97.75, 98.04, 98.08, 98.00, 98.13, 98.35, 98.30, 98.29,98.36]
y_spjbf = [99.07,98.90, 99.01, 99.06, 99.16, 99.17, 99.25, 99.29, 99.21, 99.31]
y_spjbf_dw = [99.25, 99.10, 99.17, 99.21, 99.31, 99.32, 99.40, 99.44, 99.39, 99.49]

y_err =          [0.34,0.42,0.28,0.41,0.29,0.34,0.25,0.32,0.30,0.20]
y_spjbf_err =    [0.20,0.20,0.15,0.22,0.19,0.16,0.11,0.16,0.26,0.12]
y_spjbf_dw_err = [0.15,0.18,0.13,0.14,0.14,0.14,0.09,0.13,0.21,0.09]
y_err_up = [y[i] + (y_err[i] / 2) for i in range(10)]
y_err_down = [y[i] - (y_err[i] / 2) for i in range(10)]
y_spjbf_dw_err_up = [y_spjbf_dw[i] + (y_spjbf_dw_err[i] / 2) for i in range(10)]
y_spjbf_dw_err_down = [y_spjbf_dw[i] - (y_spjbf_dw_err[i] / 2) for i in range(10)]
# 设置图框的大小
fig = plt.figure(figsize=(7,7))

# 绘图--阅读人数趋势
plt.plot(x, # x轴数据
         y, # y轴数据
         linestyle = '-', # 折线类型
         linewidth = 1, # 折线宽度
         color = 'orange', # 折线颜色
         marker = 'd', # 点的形状
         # markersize = 6, # 点的大小
        clip_on=False,
         label = 'SRFDCN without SPFNet') # 添加标签

# plt.plot(x, # x轴数据
#          y_spjbf, # y轴数据
#          linestyle = '-', # 折线类型
#          linewidth = 2, # 折线宽度
#          color = 'aqua', # 折线颜色
#          marker = '*', # 点的形状
#          # markersize = 6, # 点的大小
#
#          label = '2') # 添加标签

plt.plot(x, # x轴数据
         y_spjbf_dw, # y轴数据
         linestyle = '-', # 折线类型
         linewidth = 1, # 折线宽度
         color = 'aqua', # 折线颜色
         marker = '^', # 点的形状
         # markersize = 6, # 点的大小
         clip_on=False,
         label = 'SRFDCN') # 添加标签

# 添加标题和坐标轴标签
# plt.title('uuu')

#设置横纵坐标的名称以及对应字体格式
font2 = {'family' : 'Times New Roman',
         'weight' : 'normal',
         'size'   : 20,
         }

plt.xlabel('Number of fused superpixels',font2)
plt.ylabel('Overall accuracy', font2)
plt.xticks(fontproperties = 'Times New Roman', size = 15)
plt.yticks(fontproperties = 'Times New Roman', size = 15)
plt.grid(alpha=0.2)  # 生成网格
# 获取图的坐标信息
# 用ax=plt.gca()获得axes对象

ax = plt.gca()
'''
xmax：指定 100% 对应原始数据的值，默认值是 100，由于我们的数据是 0~1 之间的小数，所以这里要设置为 1，即 data 中的 1 表示 100%；
decimals：指定显示小数点后多少位，默认是由函数自动确定，这里我们设置成 1，使之仅显示小数点后 1 位。
'''
ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=100, decimals=1))
ax.set_xlim(0,9)
ax.set_ylim(97.5,100)
ax.fill_between(x, y_err_up,  y_err_down, facecolor='orange', alpha=0.2)
ax.fill_between(x, y_spjbf_dw_err_up,  y_spjbf_dw_err_down, facecolor='aqua', alpha=0.2)
# plt.xticks(x, [1683,   2276, 2597,  3062,  3506,  3934,  4201,  4391,  4474, 4680, 4768])
# 显示图例

plt.legend(loc='upper left', prop={'family' : 'Times New Roman', 'size'  : 20})
# 显示图形
plt.show()

