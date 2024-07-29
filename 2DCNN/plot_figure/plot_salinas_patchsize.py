
import matplotlib as mpl
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import ticker

x = ["15", "17", "19", "21", "23"]

y_2dcnn = [97.64, 98.43, 98.57, 98.92, 98.18]
y_dbda = [96.16, 97.24, 97.51, 97.50, 97.86]
y_3dcnn = [91.71, 90.80, 91.26, 93.70, 93.44]
y_hybridsn = [98.91, 99.17, 98.83, 99.34, 99.15]
y_dcn = [97.73, 98.29, 98.47, 98.83, 98.63]
y_srfdcn = [99.66, 99.50, 99.60, 99.63, 99.27]

y_2dcnn_err = [0.49, 0.18, 0.40, 0.38, 1.66]
y_dbda_err = [0.50, 1.07, 1.69, 0.80, 0.84]
y_3dcnn_err = [1.00, 2.42, 1.26, 0.22, 0.54]
y_hybridsn_err = [0.50, 0.19, 0.65, 0.29, 0.33]
y_dcn_err = [0.32, 0.48, 0.13, 0.27, 0.35]
y_srfdcn_err = [0.02, 0.19, 0.16, 0.09, 0.21]


y_2dcnn_err_up = [y_2dcnn[i] + (y_2dcnn_err[i] / 2) for i in range(5)]
y_2dcnn_err_down = [y_2dcnn[i] - (y_2dcnn_err[i] / 2) for i in range(5)]

y_dbda_err_up = [y_dbda[i] + (y_dbda_err[i] / 2) for i in range(5)]
y_dbda_err_down = [y_dbda[i] - (y_dbda_err[i] / 2) for i in range(5)]

y_3dcnn_err_up = [y_3dcnn[i] + (y_3dcnn_err[i] / 2) for i in range(5)]
y_3dcnn_err_down = [y_3dcnn[i] - (y_3dcnn_err[i] / 2) for i in range(5)]


y_hybridsn_err_up = [y_hybridsn[i] + (y_hybridsn_err[i] / 2) for i in range(5)]
y_hybridsn_err_down = [y_hybridsn[i] - (y_hybridsn_err[i] / 2) for i in range(5)]

y_dcn_err_up = [y_dcn[i] + (y_dcn_err[i] / 2) for i in range(5)]
y_dcn_err_down = [y_dcn[i] - (y_dcn_err[i] / 2) for i in range(5)]

y_srfdcn_err_up = [y_srfdcn[i] + (y_srfdcn_err[i] / 2) for i in range(5)]
y_srfdcn_err_down = [y_srfdcn[i] - (y_srfdcn_err[i] / 2) for i in range(5)]




# 设置图框的大小
fig = plt.figure(figsize=(7,7))

# 绘图--阅读人数趋势
# plt.plot(x, y_epf,linestyle = '-', linewidth = 1, color = '#50c48f', marker = '3',clip_on=False, label = 'EPF')
plt.plot(x, y_2dcnn,linestyle = '-', linewidth = 1, color = '#26ccd8', marker = 'h',clip_on=False, label = '2DCNN')
plt.plot(x, y_dbda,linestyle = '-', linewidth = 1, color = '#3685fe', marker = 's',clip_on=False, label = 'DBDA')

plt.plot(x, y_3dcnn,linestyle = '-', linewidth = 1, color = '#9977ef', marker = '^',clip_on=False, label = '3DCNN')
plt.plot(x, y_hybridsn,linestyle = '-', linewidth = 1, color = '#f5616f', marker = '*',clip_on=False, label = 'HybridSN')
plt.plot(x, y_dcn,linestyle = '-', linewidth = 1, color = '#f7b13f', marker = 'o',clip_on=False, label = 'DCNet')

# plt.plot(x, y_ssfcncrf,linestyle = '-', linewidth = 1, color = '#0780cf', marker = '>',clip_on=False, label = 'SSFCNCRF')
# plt.plot(x, y_hms,linestyle = '-', linewidth = 1, color = '#f47a75', marker = '+',clip_on=False, label = 'HMS')
plt.plot(x, y_srfdcn,linestyle = '-', linewidth = 1, color = '#ef4464', marker = 'd',clip_on=False, label = 'SRFDCN')

# 添加标题和坐标轴标签
# plt.title('uuu')

#设置横纵坐标的名称以及对应字体格式
font2 = {'family' : 'Times New Roman',
         'weight' : 'normal',
         'size'   : 15,}

plt.xlabel('Patch size',font2)
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
ax.set_xlim(0,4)
ax.set_ylim(85,100)
# ax.fill_between(x, y_epf_err_up,  y_epf_err_down, facecolor='#50c48f', alpha=0.4)
ax.fill_between(x, y_2dcnn_err_up,  y_2dcnn_err_down, facecolor='#26ccd8', alpha=0.4)
ax.fill_between(x, y_dbda_err_up,  y_dbda_err_down, facecolor='#3685fe', alpha=0.4)

ax.fill_between(x, y_3dcnn_err_up,  y_3dcnn_err_down, facecolor='#9977ef', alpha=0.4)
ax.fill_between(x, y_hybridsn_err_up,  y_hybridsn_err_down, facecolor='#f5616f', alpha=0.4)
ax.fill_between(x, y_dcn_err_up,  y_dcn_err_down, facecolor='#f7b13f', alpha=0.4)

# ax.fill_between(x, y_ssfcncrf_err_up,  y_ssfcncrf_err_down, facecolor='#0780cf', alpha=0.4)
# ax.fill_between(x, y_hms_err_up,  y_hms_err_down, facecolor='#f47a75', alpha=0.4)
ax.fill_between(x, y_srfdcn_err_up,  y_srfdcn_err_down, facecolor='#ef4464', alpha=0.4)

# plt.xticks(x, [1683,   2276, 2597,  3062,  3506,  3934,  4201,  4391,  4474, 4680, 4768])
# 显示图例

plt.legend(loc='lower right', prop={'family' : 'Times New Roman', 'size'  : 15})
# 显示图形
plt.show()
