
import matplotlib as mpl
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import ticker

x = ["6462",   "7494", "7669",  "8169",  "9891",  "9944",  "10029",  "10266",  "10949", "13115"]
y = [88.16,88.41,88.71,87.70,89.17, 89.29,88.86,87.98,88.77,89.58]
y_spjbf = [89.20 ,89.60,90.32 ,88.89, 90.73,90.58,90.35,89.24,89.79, 90.91]
y_spjbf_dw = [89.63,90.02,90.68,89.05,91.19,91.20,90.49,89.23,89.82,91.54]

y_err =          [0.17,0.92,1.39,0.27,0.45,0.59,1.11,1.02,0.09,0.40]
y_spjbf_err =    [0.06,1.17,1.58,0.41,1.10,0.68,1.16,1.12,0.55,0.62]
y_spjbf_dw_err = [0.08,0.93,1.65,0.56,0.85,0.74,1.28,0.99,0.89,0.44]

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
ax.set_ylim(87,100)
ax.fill_between(x, y_err_up,  y_err_down, facecolor='orange', alpha=0.2)
ax.fill_between(x, y_spjbf_dw_err_up,  y_spjbf_dw_err_down, facecolor='aqua', alpha=0.2)
# plt.xticks(x, [1683,   2276, 2597,  3062,  3506,  3934,  4201,  4391,  4474, 4680, 4768])
# 显示图例
plt.legend(loc='upper left', prop={'family' : 'Times New Roman', 'size'  : 20})
# 显示图形
plt.show()

