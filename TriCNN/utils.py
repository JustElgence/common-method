"""
Created on Mon Feb  7 09:21:37 2022

@author: malkhatib
"""
import scipy.io as sio
import os
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, cohen_kappa_score
import numpy as np
from operator import truediv
import random 
from sklearn.utils import shuffle

def loadData(name):
    data_path = os.path.join(os.getcwd(),'data')
    folder = '/home/t1/DL_Code/DATASET/'

    if name == 'IP':
        data = sio.loadmat(os.path.join(folder, 'IndianPines/Indian_pines_corrected.mat'))['indian_pines_corrected']
        labels = sio.loadmat(os.path.join(folder, 'IndianPines/Indian_pines.mat'))['indian_pines_gt']
    elif name == 'SA':
        data = sio.loadmat(os.path.join(folder, 'Salinas/Salinas_corrected.mat'))['salinas_corrected']
        labels = sio.loadmat(os.path.join(folder, 'Salinas/Salinas_gt.mat'))['salinas_gt']
    elif name == 'PU':
        data = sio.loadmat(os.path.join(folder, 'PaviaU/PaviaU.mat'))['paviaU']
        # data = sio.loadmat('/home/t1/DL_Code/DATASET/add_fog_dataset.mat')['Iw']
        labels = sio.loadmat(os.path.join(folder, 'PaviaU/PaviaU_gt.mat'))['paviaU_gt']
    elif name == 'PC':
        data = sio.loadmat(os.path.join(folder, 'PaviaC/Pavia.mat'))['pavia']
        labels = sio.loadmat(os.path.join(folder, 'PaviaC/Pavia_gt.mat'))['pavia_gt']
    elif name == 'BO':
        data = sio.loadmat(os.path.join(folder, 'Botswana.mat'))['Botswana']
        labels = sio.loadmat(os.path.join(folder, 'Botswana_gt.mat'))['Botswana_gt']
    elif name == 'GP':
        data = sio.loadmat(os.path.join(folder, 'Gulfport.mat'))['gulfport']
        labels = sio.loadmat(os.path.join(folder, 'Gulfport_gt.mat'))['gulfport_gt']
    elif name == "HU":
        img = sio.loadmat('/home/t1/DL_Code/DATASET/houston2013/' + 'Houston.mat')
        data = img['Houston']
        labels = sio.loadmat('/home/t1/DL_Code/DATASET/houston2013/' + 'Houston_gt.mat')['houston_gt']
    return data, labels

def splitTrainTestSet(X, y, testRatio, randomState=345):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=testRatio, random_state=randomState,
                                                        stratify=y)
    return X_train, X_test, y_train, y_test


def splitTrainTestSet1(X, y, testRatio, randomState=345):
    train_indices, test_indices = [], []
    Y_train_indices, Y_test_indices = [], []
    train_size = np.ones(16)*10


    for c in np.unique(y):
        if c == 0:
            continue 
        c = int(c)
        indices = np.nonzero(y == c)
        X = list(zip(*indices))
        print(train_size[c],c)

        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=int(train_size[c]))
        train_indices += X_train
        test_indices += X_test
        Y_train_indices += y_train
        Y_test_indices += y_test
    
    X_train_idx = [list(t) for t in zip(*train_indices)]
    X_test_idx = [list(t) for t in zip(*test_indices)]
    y_train_idx = [list(t) for t in zip(*Y_train_indices)]
    y_test_idx = [list(t) for t in zip(*Y_test_indices)]

    X_train_data = np.zeros_like(X)
    X_test_data = np.zeros_like(X)
    y_train_label = np.zeros_like(y)
    y_test_label = np.zeros_like(y)

    X_train_data[tuple(X_train_idx)] = X[tuple(X_train_idx), :, : , :]
    X_test_data[tuple(X_test_idx)] = X[tuple(X_test_idx), :, : , :]
    
    y_train_label[tuple(y_train_idx)] = y[tuple(y_train_idx)]
    y_test_label[tuple(y_test_idx)] = y[tuple(y_test_idx)]

    return X_train_data, X_test_data, y_train_label, y_test_label

def splitTrainTestSet2(X, y, testRatio, randomState=345):

    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=testRatio, random_state=randomState,
    #                                                     stratify=y)
    train_size = np.ones(15)*10
    window_size = 9
    X_train_indices, X_test_indices, Y_train_indices, Y_test_indices = [], [], [], []
    for c in np.unique(y):
        c = int(c)

        # if c == 0:
        #     continue

        indices = np.nonzero(y == c)

        y_temp = list(zip(*indices))
        print(train_size[c],c)
        y_train, y_test = train_test_split(y_temp, train_size = int(train_size[c]))

        Y_train_indices += y_train
        Y_test_indices += y_test

        #*转置
    X_train_idx = [list(t) for t in zip(*Y_train_indices)]
    X_test_idx = [list(t) for t in zip(*Y_test_indices)]
    y_train_idx = [list(t) for t in zip(*Y_train_indices)]
    y_test_idx = [list(t) for t in zip(*Y_test_indices)]

    X_train_data = np.zeros_like(X)
    X_test_data = np.zeros_like(X)
    y_train_label = np.zeros_like(y)
    y_test_label = np.zeros_like(y)

    X_train_data[tuple(X_train_idx)] = X[tuple(X_train_idx), :, : , :]
    X_test_data[tuple(X_test_idx)] = X[tuple(X_test_idx), :, : , :]
    
    y_train_label[tuple(y_train_idx)] = y[tuple(y_train_idx)]
    y_test_label[tuple(y_test_idx)] = y[tuple(y_test_idx)]

    X_train_data = X_train_data[y_train_label > 0, :, :, :]
    X_test_data = X_test_data[y_test_label > 0, :, :, :]
    y_train_label = y_train_label[y_train_label > 0]
    y_test_label = y_test_label[y_test_label > 0]

    return X_train_data, X_test_data, y_train_label, y_test_label


def splitTrainTestSet_backup(X, y, testRatio, randomState=345):
    #    train_size = [100,10,10,10,10,10, 10,10,10,10]
       #train_size = [100,500,500,68,500,500,451,26,500,500,500,500,151,500,500,500,500,14,500,500,500]
    #    train_size = [100,5,5,5,5,5, 5,5,5,5]
    train_size = [100,5,5,5,5,5, 5,5,5,5,5,5,5 ,5 ,5 ,5,5]
    #    train_size = [100,1,1,1,1,1, 1,1,1,1]
    #    train_size = [100,2,2,2,2,2, 2,2,2,2]
    #    train_size = [100,3,3,3,3,3, 3,3,3,3]
    #    train_size = [100,4,4,4,4,4, 4,4,4,4]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=testRatio, random_state=randomState,
                                                        stratify=y)
    return X_train, X_test, y_train, y_test


def applyPCA(X, numComponents=75):
    newX = np.reshape(X, (-1, X.shape[2]))
    pca = PCA(n_components=numComponents, whiten=True)
    newX = pca.fit_transform(newX)
    newX = np.reshape(newX, (X.shape[0],X.shape[1], numComponents))
    return newX, pca

def padWithZeros(X, margin=2):
    newX = np.zeros((X.shape[0] + 2 * margin, X.shape[1] + 2* margin, X.shape[2]))
    x_offset = margin
    y_offset = margin
    newX[x_offset:X.shape[0] + x_offset, y_offset:X.shape[1] + y_offset, :] = X
    return newX

def createImageCubes(X, y, windowSize=5, removeZeroLabels = True):
    margin = int((windowSize - 1) / 2)
    zeroPaddedX = padWithZeros(X, margin=margin)
    # split patches
    patchesData = np.zeros((X.shape[0] * X.shape[1], windowSize, windowSize, X.shape[2]))
    patchesLabels = np.zeros((X.shape[0] * X.shape[1]))
    patchIndex = 0
    for r in range(margin, zeroPaddedX.shape[0] - margin):
        for c in range(margin, zeroPaddedX.shape[1] - margin):
            patch = zeroPaddedX[r - margin:r + margin + 1, c - margin:c + margin + 1]   
            patchesData[patchIndex, :, :, :] = patch
            patchesLabels[patchIndex] = y[r-margin, c-margin]
            patchIndex = patchIndex + 1
    if removeZeroLabels:
        patchesData = patchesData[patchesLabels>0,:,:,:]
        patchesLabels = patchesLabels[patchesLabels>0]
        patchesLabels -= 1
    return patchesData, patchesLabels

def AA_andEachClassAccuracy(confusion_matrix):
    counter = confusion_matrix.shape[0]
    list_diag = np.diag(confusion_matrix)
    list_raw_sum = np.sum(confusion_matrix, axis=1)
    each_acc = np.nan_to_num(truediv(list_diag, list_raw_sum))
    average_acc = np.mean(each_acc)
    return each_acc, average_acc


def target(name):
    if name == 'IP':
        target_names = ['Alfalfa', 'Corn-notill', 'Corn-mintill', 'Corn'
                        ,'Grass-pasture', 'Grass-trees', 'Grass-pasture-mowed', 
                        'Hay-windrowed', 'Oats', 'Soybean-notill', 'Soybean-mintill',
                        'Soybean-clean', 'Wheat', 'Woods', 'Buildings-Grass-Trees-Drives',
                        'Stone-Steel-Towers']
    elif name == 'SA':
        target_names = ['Brocoli_green_weeds_1','Brocoli_green_weeds_2','Fallow','Fallow_rough_plow','Fallow_smooth',
                        'Stubble','Celery','Grapes_untrained','Soil_vinyard_develop','Corn_senesced_green_weeds',
                        'Lettuce_romaine_4wk','Lettuce_romaine_5wk','Lettuce_romaine_6wk','Lettuce_romaine_7wk',
                        'Vinyard_untrained','Vinyard_vertical_trellis']
    elif name == 'PU':
        target_names = ['Asphalt','Meadows','Gravel','Trees', 'Painted metal sheets','Bare Soil','Bitumen',
                        'Self-Blocking Bricks','Shadows']
    elif name == 'BO':
        target_names = ['Water','Hippo_grass','Floodpain_grasses_1','Floodpain_grasses_2','Reeds','Riparian','Firescar',
                        'Island_interior','Acacia_woodlands','Acacia_shrublands','Acacia_grasslands','Short_mopane',
                        'Mixed_mopane','Exposed_soils']
    elif name == 'GP':
        target_names = ['Tree', 'Shadow', 'Grass', 'Dead Grass', 'Asphalt', 'Dirt']
        
    return target_names 
    
def num_classes(dataset):
    if dataset == 'PU' or dataset == 'PC':
        output_units = 9
    elif dataset == 'IP' or dataset == 'SA':
        output_units = 16
    elif dataset == 'BO':
        output_units = 14
    elif dataset == 'HU':
        output_units = 15
    return output_units




def Patch(data,height_index,width_index, PATCH_SIZE):
    height_slice = slice(height_index, height_index+PATCH_SIZE)
    width_slice = slice(width_index, width_index+PATCH_SIZE)
    patch = data[height_slice, width_slice, :]
    
    return patch


