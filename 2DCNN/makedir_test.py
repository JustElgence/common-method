import os

b = os.getcwd() + '\\result\\'
if not os.path.exists(b):  # 判断当前路径是否存在，没有则创建new文件夹
    os.makedirs(b)
# mode 模式
# w 只能操作写入  r 只能读取 a 向文件追加
# w+ 可读可写 r+可读可写  a+可读可追加
# wb+写入进制数据
# w模式打开文件，如果而文件中有数据，再次写入内容，会把原来的覆盖掉
# patch_size = hyperparams['patch_size']
# model = hyperparams['model']
# dataset_name = hyperparams['dataset']
patch_size = '23'
model = '2dcnn'
dataset_name = 'PaviaU'

file_name = b + "classification_report_" + dataset_name + "_" + model + "_" + patch_size + ".txt"
print(file_name)

file_handle = open(file_name, mode='a')
# # \n 换行符
text = ""
text += "---\n"
text += "666\n"
text += "222\n"
text += "6666\n"
file_handle.write(text)
file_handle.close()
print('ok')