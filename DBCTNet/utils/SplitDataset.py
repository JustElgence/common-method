from sklearn.model_selection import train_test_split
from collections import Counter
import numpy as np

def split_dataset(x,y,train_ratio=0.1):
    data_remove_zero = x[y>0,:,:,:]
    label_remove_zero = y[y>0]
    label_remove_zero -= 1
    indices = np.where(y > 0)
    test_ratio = 1 - 2*train_ratio
    x_train,x_test,y_train,y_test = train_test_split(data_remove_zero,label_remove_zero,
                                                     stratify=label_remove_zero,
                                                     test_size=test_ratio)
    x_train,x_val,y_train,y_val = train_test_split(x_train,y_train,
                                                 stratify=y_train,
                                                 test_size=0.5)

    classes = int(np.max(y_train))+1
    total_w = len(y_train)/classes
    class_map = Counter(y_train)#0-8                            
    sampler = [total_w/class_map[i] for i in range(classes)]

    return x_train,y_train,x_val,y_val,x_test,y_test,sampler

def split_dataset_draw(x,y,train_ratio=0.1):
    data_remove_zero = x[y>0,:,:,:]
    label_remove_zero = y[y>0]
    label_remove_zero -= 1
    indices = np.argwhere(y > 0)
    test_ratio = 1 - 2*train_ratio
    x_train,x_test,y_train,y_test,x_indices,y_indices = train_test_split(data_remove_zero,label_remove_zero,indices,
                                                     stratify=label_remove_zero,
                                                     test_size=test_ratio)
    x_train,x_val,y_train,y_val = train_test_split(x_train,y_train,
                                                 stratify=y_train,
                                                 test_size=0.5)

    classes = int(np.max(y_train))+1
    total_w = len(y_train)/classes
    class_map = Counter(y_train)#0-8                            
    sampler = [total_w/class_map[i] for i in range(classes)]

    return x_train,y_train,x_val,y_val,x_test,y_test,sampler,y_indices

def split_dataset_bynum(x, y, train_ratio = 5):

    data_remove_zero = x[y>0,:,:,:]
    label_remove_zero = y[y>0]
    label_remove_zero -= 1

    x_train,x_test,y_train,y_test = split_dataset_bynum1(data_remove_zero, label_remove_zero, train_num=train_ratio * 2)

    x_train,x_val,y_train,y_val = split_dataset_bynum1(x_train, y_train,train_num=train_ratio)

    x_train = x_train[y_train>=0,:,:,:]
    y_train = y_train[y_train>=0]

    x_val = x_val[y_val>=0,:,:,:]
    y_val = y_val[y_val>=0]

    x_test = x_test[y_test>=0]
    y_test = y_test[y_test>=0]

    classes = int(np.max(y_train)) + 1
    total_w = len(y_train)/classes
    class_map = Counter(y_train)                            
    sampler = [total_w/class_map[i] for i in range(classes)]

    return x_train,y_train,x_val,y_val,x_test,y_test,sampler

def split_dataset_bynum1(x, y, train_num = 5):


    data_train_gt = np.full_like(x, fill_value=-1)
    data_test_gt = np.full_like(x, fill_value=-1)
    
    train_gt = np.full_like(y, fill_value=-1)
    test_gt = np.full_like(y, fill_value=-1)

    train_indices, test_indices = [], []
    train_size = np.ones(16) * train_num

    for c in np.unique(y):
        if c >= 0:
            c = int(c)
            indices = np.nonzero(y == c)
            X = list(zip(*indices)) 
            print(train_size[c],c)

            train, test = train_test_split(X, train_size=int(train_size[c]))
            train_indices += train
            test_indices += test
       
    train_indices = [list(t) for t in zip(*train_indices)]
    test_indices = [list(t) for t in zip(*test_indices)]

    data_train_gt[tuple(train_indices),:,:,:] = x[tuple(train_indices),:,:,:]
    data_test_gt[tuple(test_indices),:,:,:] = x[tuple(test_indices),:,:,:]
    train_gt[tuple(train_indices)] = y[tuple(train_indices)]
    test_gt[tuple(test_indices)] = y[tuple(test_indices)]

    return data_train_gt, data_test_gt, train_gt, test_gt