
import matplotlib as mpl
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import ticker

x = ["0.5", "1", "1.5", "2", "2.5"]

y_epf =      [89.28, 91.42, 91.89, 92.19, 92.53]
y_2dcnn =    [96.28, 98.92, 99.14, 99.56, 99.63]
y_dbda =     [93.39, 97.50, 98.57, 99.21, 99.24]
y_3dcnn =    [85.50, 93.70, 93.72, 95.98, 95.96]
y_hybridsn = [97.30, 99.34, 99.52, 99.58, 99.76]
y_dcn =      [96.92, 98.83, 99.28, 99.52, 99.71]
y_ssfcncrf = [84.05, 95.87, 96.94, 97.59, 98.15]
y_hms =      [99.18, 99.36, 99.51, 99.55, 99.52]
y_srfdcn =   [98.75, 99.63, 99.60, 99.68, 99.69]

y_epf_err =   [1.46, 0.16, 0.25, 0.19, 0.29]
y_2dcnn_err = [0.50, 0.38, 0.68, 0.09, 0.18]
y_dbda_err =  [2.35, 0.80, 0.17, 0.09, 0.06]
y_3dcnn_err = [2.38, 0.22, 0.50, 0.20, 0.27]
y_hybridsn_err = [0.08, 0.29, 0.11, 0.10, 0.04]
y_dcn_err =      [0.78, 0.27, 0.07, 0.13, 0.06]
y_ssfcncrf_err = [4.07, 0.63, 0.55, 0.67, 0.17]
y_hms_err =      [0.26, 0.10, 0.10, 0.07, 0.08]
y_srfdcn_err =   [0.34, 0.09, 0.11, 0.07, 0.06]


y_epf_err_up = [y_epf[i] + (y_epf_err[i] / 2) for i in range(5)]
y_epf_err_down = [y_epf[i] - (y_epf_err[i] / 2) for i in range(5)]

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

y_ssfcncrf_err_up = [y_ssfcncrf[i] + (y_ssfcncrf_err[i] / 2) for i in range(5)]
y_ssfcncrf_err_down = [y_ssfcncrf[i] - (y_ssfcncrf_err[i] / 2) for i in range(5)]

y_hms_err_up = [y_hms[i] + (y_hms_err[i] / 2) for i in range(5)]
y_hms_err_down = [y_hms[i] - (y_hms_err[i] / 2) for i in range(5)]

y_srfdcn_err_up = [y_srfdcn[i] + (y_srfdcn_err[i] / 2) for i in range(5)]
y_srfdcn_err_down = [y_srfdcn[i] - (y_srfdcn_err[i] / 2) for i in range(5)]


# 设置图框的大小
fig = plt.figure(figsize=(7,7))

# 绘图--阅读人数趋势
plt.plot(x, y_epf,linestyle = '-', linewidth = 1, color = '#50c48f', marker = '3',clip_on=False, label = 'EPF')
plt.plot(x, y_2dcnn,linestyle = '-', linewidth = 1, color = '#26ccd8', marker = 'h',clip_on=False, label = '2DCNN')
plt.plot(x, y_dbda,linestyle = '-', linewidth = 1, color = '#3685fe', marker = 's',clip_on=False, label = 'DBDA')

plt.plot(x, y_3dcnn,linestyle = '-', linewidth = 1, color = '#9977ef', marker = '^',clip_on=False, label = '3DCNN')
plt.plot(x, y_hybridsn,linestyle = '-', linewidth = 1, color = '#f5616f', marker = '*',clip_on=False, label = 'HybridSN')
plt.plot(x, y_dcn,linestyle = '-', linewidth = 1, color = '#f7b13f', marker = 'o',clip_on=False, label = 'DCNet')

plt.plot(x, y_ssfcncrf,linestyle = '-', linewidth = 1, color = '#0780cf', marker = '>',clip_on=False, label = 'SSFCNCRF')
plt.plot(x, y_hms,linestyle = '-', linewidth = 1, color = '#f47a75', marker = '+',clip_on=False, label = 'HMS')
plt.plot(x, y_srfdcn,linestyle = '-', linewidth = 1, color = '#ef4464', marker = 'd',clip_on=False, label = 'SRFDCN')

# 添加标题和坐标轴标签
# plt.title('uuu')

#设置横纵坐标的名称以及对应字体格式
font2 = {'family' : 'Times New Roman',
         'weight' : 'normal',
         'size'   : 15,}

plt.xlabel('Number of training sample (%)',font2)
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
ax.set_ylim(80,100)
ax.fill_between(x, y_epf_err_up,  y_epf_err_down, facecolor='#50c48f', alpha=0.4)
ax.fill_between(x, y_2dcnn_err_up,  y_2dcnn_err_down, facecolor='#26ccd8', alpha=0.4)
ax.fill_between(x, y_dbda_err_up,  y_dbda_err_down, facecolor='#3685fe', alpha=0.4)

ax.fill_between(x, y_3dcnn_err_up,  y_3dcnn_err_down, facecolor='#9977ef', alpha=0.4)
ax.fill_between(x, y_hybridsn_err_up,  y_hybridsn_err_down, facecolor='#f5616f', alpha=0.4)
ax.fill_between(x, y_dcn_err_up,  y_dcn_err_down, facecolor='#f7b13f', alpha=0.4)

ax.fill_between(x, y_ssfcncrf_err_up,  y_ssfcncrf_err_down, facecolor='#0780cf', alpha=0.4)
ax.fill_between(x, y_hms_err_up,  y_hms_err_down, facecolor='#f47a75', alpha=0.4)
ax.fill_between(x, y_srfdcn_err_up,  y_srfdcn_err_down, facecolor='#ef4464', alpha=0.4)

# plt.xticks(x, [1683,   2276, 2597,  3062,  3506,  3934,  4201,  4391,  4474, 4680, 4768])
# 显示图例

plt.legend(loc='lower right', prop={'family' : 'Times New Roman', 'size'  : 15})
# 显示图形
plt.show()

