# -*- coding: utf-8 -*-
# @Auther   : Mingsong Li (lms-07)
# @Time     : 2023-Apr
# @Address  : Time Lab @ SDU
# @FileName : process_dl.py
# @Project  : AMS-M2ESL (HSIC), IEEE TGRS

# for IP and UP data sets, main processing file for the proposed AMS-M2ESL model

import os
import time
import torch
import random
import numpy as np
from sklearn import metrics
from ptflops import get_model_complexity_info

import utils.evaluation as evaluation
import utils.data_load_operate as data_load_operate
import visual.cls_visual as cls_visual
import model.AMS_M2ESL as AMS_M2ESL
import visdom
import seaborn as sns
# import utils.data_load_operate_AIPS as data_load_operate


time_current = time.strftime("%y-%m-%d-%H.%M", time.localtime())

# random seed setting
seed = 20

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)  # Numpy module.
random.seed(seed)  # Python random module.
torch.manual_seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def convert_to_color_(arr_2d, palette=None):
    """Convert an array of labels to RGB color-encoded image.

    Args:
        arr_2d: int 2D array of labels
        palette: dict of colors used (label number -> RGB tuple)

    Returns:
        arr_3d: int 2D images of color-encoded labels in RGB format

    """
    arr_3d = np.zeros((arr_2d.shape[0], arr_2d.shape[1], 3), dtype=np.uint8)
    if palette is None:
        raise Exception("Unknown color palette")
    print(palette)
    # quit()
    for c, i in palette.items():
        m = arr_2d == c
        arr_3d[m] = i

    return arr_3d

def convert_to_color(x, N_Classes, palette):
    
    if palette is None:
    # Generate color palette
        palette = {0: (0, 0, 0)}

    for k, color in enumerate(sns.color_palette("hls", N_Classes - 1)):
        palette[k + 1] = tuple(np.asarray(255 * np.array(color), dtype='uint8'))

    invert_palette = {v: k for k, v in palette.items()}
    
    return convert_to_color_(x, palette=palette)

# viz = visdom.Visdom(env = "Draw")

# if not viz.check_connection:
# 	print("Visdom is not connected. Did you run 'python -m visdom.server' ?")


###                0    ###
model_list = ['AMS-M2ESL']
model_flag = 0
model_spa_set = {0}
model_spe_set = {}
model_spa_spe_set = {}
model_3D_spa_set = {}
model_3D_spa_flag = 0

last_batch_flag = 1

if model_flag in model_spa_set:
    model_type_flag = 1
    if model_flag in model_3D_spa_set:
        model_3D_spa_flag = 1
elif model_flag in model_spe_set:
    model_type_flag = 2
elif model_flag in model_spa_spe_set:
    model_type_flag = 3

# 0-1
data_set_name_list = ['IP', 'PaviaU', "Salinas", "PaviaC"]
data_set_name = data_set_name_list[1]
data_set_name = 'PaviaC'
data_set_path = os.path.join(os.getcwd(), 'data')

# control running times
# seed_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
# seed_list=[0,1,2,3,4]
# seed_list=[0,1,2]
# seed_list=[0,1]
seed_list = [0]

# data set split
flag_list = [0, 1]  # ratio or num

if data_set_name == 'IP':
    ratio_list = [0.025, 0.005]
    ratio = 2.5
elif data_set_name == 'PaviaU':
    ratio_list = [0.01, 0.001]
    ratio = 1.0
elif data_set_name == 'Salinas':
    ratio_list = [0.025, 0.001]
    ratio = 1.0
elif data_set_name == 'PaviaC':
    ratio_list = [0.005, 0.001]
    ratio = 1.0
elif data_set_name == 'WHU-Hi-LongKou':
    ratio_list = [0.005, 0.001]
    ratio = 0.005
elif data_set_name == 'KSC':
    ratio_list = [0.01, 0.001]
    ratio = 0.005
elif data_set_name == 'Botswana':
    ratio_list = [0.01, 0.001]
    ratio = 0.005
    
num_list = [50, 0]  # [train_num,val_num]

patch_size = 9
patch_length = 4

results_save_path = \
    os.path.join(os.getcwd(), 'output/results', model_list[model_flag] + str("_") +
                 data_set_name + str("_") + str(time_current) + str("_seed_") + str(seed) + str("_ratio_") + str(
        ratio) + str("_patch_size_") + str(patch_size))
cls_map_save_path = \
    os.path.join(os.path.join(os.getcwd(), 'output/cls_maps'), model_list[model_flag] + str("_") +
                 data_set_name + str("_") + str(time_current) + str("_seed_") + str(seed) + str("_ratio_") + str(ratio))

if __name__ == '__main__':

    data, gt = data_load_operate.load_data(data_set_name, data_set_path)
    data = data_load_operate.standardization(data)

    # dr=math.ceil(data.shape[-1]*0.3)
    # data=data_load_operate.applyPCA(data,dr) # for abla of MNF
    data = data_load_operate.HSI_MNF(data, MNF_ratio=0.3)

    gt_reshape = gt.reshape(-1)
    height, width, channels = data.shape
    class_count = max(np.unique(gt))

    batch_size = 64
    max_epoch = 100
    learning_rate = 0.001
    loss = torch.nn.CrossEntropyLoss()

    OA_ALL = []
    AA_ALL = []
    KPP_ALL = []
    EACH_ACC_ALL = []
    Train_Time_ALL = []
    Test_Time_ALL = []
    CLASS_ACC = np.zeros([len(seed_list), class_count])

    # data pad zero
    # data:[h,w,c]->data_padded:[h+2l,w+2l,c]
    data_padded = data_load_operate.data_pad_zero(data, patch_length)
    height_patched, width_patched = data_padded.shape[0], data_padded.shape[1]

    # data_total_index = np.arange(data.shape[0] * data.shape[1])  # For total sample cls_map.

    for run in range(0, 10):
        tic1 = time.perf_counter()
        train_data_index, test_data_index, all_data_index = data_load_operate.sampling(ratio_list,
                                                                                       num_list,
                                                                                       gt_reshape,
                                                                                       class_count,
                                                                                       flag_list[0])
        index = (train_data_index, test_data_index)

        train_iter, test_iter = data_load_operate.generate_iter(data_padded, height, width,
                                                                gt_reshape, index, patch_length,
                                                                batch_size,
                                                                model_type_flag,
                                                                model_3D_spa_flag,
                                                                last_batch_flag)

        # load data for the cls map of the total samples
        # total_iter = data_load_operate.generate_iter_total(data_padded, height, width, gt_reshape, data_total_index, patch_length,
        #                                                batch_size, model_type_flag, model_3D_spa_flag)

        if model_flag == 0:
            net = AMS_M2ESL.AMS_M2ESL_(in_channels=channels, patch_size=patch_size, num_classes=class_count,
                                        ds=data_set_name)

        net.to(device)

        # efficiency test, model complexity and computational cost
        # flops,para=get_model_complexity_info(net,(channels,1,1),as_strings=False,print_per_layer_stat=True, verbose=True)
        # flops,para=get_model_complexity_info(net,(patch_size,patch_size,channels),as_strings=False,print_per_layer_stat=True, verbose=True)
        # # # flops,para=get_model_complexity_info(net,(1,1,patch_size,patch_size,channels),as_strings=False,print_per_layer_stat=True, verbose=True)
        # print("para(M):{:.3f},\n flops(M):{:.3f}".format(para/(1000**2),flops/(1000**2)))

        train_loss_list = [100]
        train_acc_list = [0]

        optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

        for epoch in range(max_epoch):
            train_acc_sum, trained_samples_counter = 0.0, 0
            batch_counter, train_loss_sum = 0, 0
            time_epoch = time.time()

            if model_type_flag == 1:  # data for single spatial net
                for X_spa, y in train_iter:
                    X_spa, y = X_spa.to(device), y.to(device)
                    y_pred = net(X_spa)

                    ls = loss(y_pred, y.long())

                    optimizer.zero_grad()
                    ls.backward()
                    optimizer.step()

                    train_loss_sum += ls.cpu().item()
                    train_acc_sum += (y_pred.argmax(dim=1) == y).sum().cpu().item()
                    trained_samples_counter += y.shape[0]
                    batch_counter += 1
                    epoch_first_iter = 0
            elif model_type_flag == 2:  # data for single spectral net
                for X_spe, y in train_iter:
                    X_spe, y = X_spe.to(device), y.to(device)
                    y_pred = net(X_spe)

                    ls = loss(y_pred, y.long())

                    optimizer.zero_grad()
                    ls.backward()
                    optimizer.step()

                    train_loss_sum += ls.cpu().item()
                    train_acc_sum += (y_pred.argmax(dim=1) == y).sum().cpu().item()
                    trained_samples_counter += y.shape[0]
                    batch_counter += 1
                    epoch_first_iter = 0
            elif model_type_flag == 3:  # data for spectral-spatial net
                for X_spa, X_spe, y in train_iter:
                    X_spa, X_spe, y = X_spa.to(device), X_spe.to(device), y.to(device)
                    y_pred = net(X_spa, X_spe)

                    ls = loss(y_pred, y.long())

                    optimizer.zero_grad()
                    ls.backward()
                    optimizer.step()

                    train_loss_sum += ls.cpu().item()
                    train_acc_sum += (y_pred.argmax(dim=1) == y).sum().cpu().item()
                    trained_samples_counter += y.shape[0]
                    batch_counter += 1
                    epoch_first_iter = 0

            torch.cuda.empty_cache()

            train_loss_list.append(train_loss_sum)
            train_acc_list.append(train_acc_sum / trained_samples_counter)

            print('epoch: %d, training_sampler_num: %d, batch_count: %.2f, train loss: %.6f, tarin loss sum: %.6f, '
                  'train acc: %.3f, train_acc_sum: %.1f, time: %.1f sec' %
                  (epoch + 1, trained_samples_counter, batch_counter, train_loss_sum / batch_counter, train_loss_sum,
                   train_acc_sum / trained_samples_counter, train_acc_sum, time.time() - time_epoch))

        toc1 = time.perf_counter()
        print('Training stage finished:\n epoch %d, loss %.4f, train acc %.3f, training time %.2f s'
              % (epoch + 1, train_loss_sum / batch_counter, train_acc_sum / trained_samples_counter, toc1 - tic1))
        training_time = toc1 - tic1
        Train_Time_ALL.append(training_time)

        print("\n\n====================Starting evaluation for testing set.========================\n")

        pred_test = []
        matrix = np.zeros((610, 340))
        # torch.cuda.empty_cache()
        with torch.no_grad():
            # net.load_state_dict(torch.load(model_save_path+"_best_model.pt"))
            net.eval()
            train_acc_sum, samples_num_counter = 0.0, 0
            tic11 = time.time()
            if model_type_flag == 1:  # data for single spatial net
                for X_spa, y, pos in test_iter:
                    X_spa = X_spa.to(device)

                    tic2 = time.perf_counter()
                    y_pred = net(X_spa)
                    toc2 = time.perf_counter()
                    temp = np.array(y_pred.cpu().argmax(axis=1)) 
                    pred_test.extend(temp)
                    # for i in range(batch_size):
                    #     row = int(pos[i] // 340)
                    #     col = int(pos[i] - row * 340)
                    #     matrix[row, col] = temp[i]
                # for X_spa, y  in test_iter:
                #     X_spa = X_spa.to(device)

                #     tic2 = time.perf_counter()
                #     y_pred = net(X_spa)
                #     toc2 = time.perf_counter()
                #     temp = np.array(y_pred.cpu().argmax(axis=1))
                #     pred_test.extend(temp)

            elif model_type_flag == 2:  # data for single spectral net
                for X_spe, y in test_iter:
                    X_spe = X_spe.to(device)

                    tic2 = time.perf_counter()
                    y_pred = net(X_spe)
                    toc2 = time.perf_counter()

                    pred_test.extend(np.array(y_pred.cpu().argmax(axis=1)))
            elif model_type_flag == 3:  # data for spectral-spatial net
                for X_spa, X_spe, y in test_iter:
                    X_spa = X_spa.to(device)
                    X_spe = X_spe.to(device)

                    tic2 = time.perf_counter()
                    y_pred = net(X_spa, X_spe)
                    toc2 = time.perf_counter()

                    pred_test.extend(np.array(y_pred.cpu().argmax(axis=1)))
            toc22 = time.time()
            y_gt = gt_reshape[test_data_index] - 1
            OA = metrics.accuracy_score(y_gt, pred_test)
            confusion_matrix = metrics.confusion_matrix(pred_test, y_gt)
            print("confusion_matrix\n{}".format(confusion_matrix))
            ECA, AA = evaluation.AA_ECA(confusion_matrix)
            kappa = metrics.cohen_kappa_score(pred_test, y_gt)




            # cls_report = evaluation.claification_report(y_gt, pred_test, data_set_name)
            # print("classification_report\n{}".format(cls_report))

            # Visualization for all the labeled samples and total the samples
            # sample_list1 = [total_iter]
            # sample_list2 = [all_iter, all_data_index]

            # Visualization.gt_cls_map(gt,cls_map_save_path)
            # cls_visual.pred_cls_map_dl(sample_list1, net, gt, cls_map_save_path, model_type_flag)
            # cls_visual.pred_cls_map_dl(sample_list2,net,gt,cls_map_save_path)

            testing_time = toc22 - tic11
            Test_Time_ALL.append(testing_time)
            OA_ALL.append(OA)
            AA_ALL.append(AA)
            KPP_ALL.append(kappa)
            EACH_ACC_ALL.append(ECA)

        torch.cuda.empty_cache()
        del net, train_iter, test_iter

    def display_predictions(pred, vis, gt=None, caption=""):
        if gt is None:

            vis.images([np.transpose(pred, (2, 0, 1))],
                        opts={'caption': caption})
        else:

            vis.images([np.transpose(pred, (2, 0, 1)),
                        np.transpose(gt, (2, 0, 1))],
                        nrow=2,
                        opts={'caption': caption})
    # display_predictions(convert_to_color(matrix, 10, palette = None), viz, caption="prediction")

    OA_ALL = np.array(OA_ALL)
    AA_ALL = np.array(AA_ALL)
    KPP_ALL = np.array(KPP_ALL)
    EACH_ACC_ALL = np.array(EACH_ACC_ALL)
    Train_Time_ALL = np.array(Train_Time_ALL)
    Test_Time_ALL = np.array(Test_Time_ALL)

    np.set_printoptions(precision=4)
    print("\n====================Mean result of {} times runs =========================".format(len(seed_list)))
    print('List of OA:', list(OA_ALL))
    print('List of AA:', list(AA_ALL))
    print('List of KPP:', list(KPP_ALL))
    print('OA=', round(np.mean(OA_ALL) * 100, 2), '+-', round(np.std(OA_ALL, ddof=1) * 100, 4))
    print('AA=', round(np.mean(AA_ALL) * 100, 2), '+-', round(np.std(AA_ALL) * 100, 2))
    print('Kpp=', round(np.mean(KPP_ALL) * 100, 2), '+-', round(np.std(KPP_ALL) * 100, 2))
    print('Acc per class=', np.mean(EACH_ACC_ALL, 0), '+-', np.std(EACH_ACC_ALL, 0))

    print("Average training time=", round(np.mean(Train_Time_ALL), 2), '+-', round(np.std(Train_Time_ALL), 3))
    print("Average testing time=", round(np.mean(Test_Time_ALL), 5), '+-', round(np.std(Test_Time_ALL), 5))
