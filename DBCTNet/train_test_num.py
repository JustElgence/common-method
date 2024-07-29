import numpy as np
import copy
import scipy.io as sio
from sklearn.metrics import confusion_matrix
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import time
from transformers import get_cosine_schedule_with_warmup
from DBCTNet import DBCTNet
from utils.FocalLoss import FocalLoss
from utils.CreateCube import create_patch
from utils.SplitDataset import split_dataset, split_dataset_bynum
parser = argparse.ArgumentParser()
parser.add_argument('--dataset',default='PU',help='dataset')
parser.add_argument('--train_ratio', default=0.01,help='ratio of train')
parser.add_argument('--epochs',default=300,help='epoch')
parser.add_argument('--lr',default=0.001,help='learning rate')
parser.add_argument('--gamma',default=0.4,help='gamma of focal loss')
parser.add_argument('--n_run',type=int, default=1,help='times to run')
args = parser.parse_args()

config={'batch_size':32,
        'dataset':args.dataset,
        'model':'DBCTNet',
        'train_ratio':float(args.train_ratio),
        'weight_decay':0.001,
        'epoch':int(args.epochs),
        'patch_size':9,
        'fc_dim':16,
        'heads':2,
        'drop':0.1,
        'lr':float(args.lr),
        'gamma':float(args.gamma),
        }

def load_data(dataset):
    if dataset == 'PU':
        data = sio.loadmat('/home/t1/DL_Code/DATASET/PaviaU/PaviaU.mat')['paviaU']
        labels = sio.loadmat('/home/t1/DL_Code/DATASET/PaviaU/PaviaU_gt.mat')['paviaU_gt']
    if dataset == 'PC':
        data = sio.loadmat('/home/t1/DL_Code/DATASET/PaviaC/Pavia.mat')['pavia']
        labels = sio.loadmat('/home/t1/DL_Code/DATASET/PaviaC/Pavia_gt.mat')['pavia_gt']
    if dataset == 'HU':
        data = sio.loadmat('/home/t1/DL_Code/DATASET/houston2013/Houston.mat')['Houston']
        labels = sio.loadmat('/home/t1/DL_Code/DATASET/houston2013/Houston_gt.mat')['houston_gt']
    if dataset == 'SA':
        data = sio.loadmat('/home/t1/DL_Code/DATASET/Salinas/Salinas_corrected.mat')['salinas_corrected']
        labels = sio.loadmat('/home/t1/DL_Code/DATASET/Salinas/Salinas_gt.mat')['salinas_gt']
    else:
        print('Please add your dataset.')
    return data, labels

def train(model,classes,config,device,sampler,data_loader):
    epochs =  config['epoch']
    lr = config['lr']
    wd = config['weight_decay']
    dataset = config['dataset']
    gamma = config['gamma']
    model_name = config['model']
    model = model.to(device)
    criterion = FocalLoss(class_num=classes,gamma=gamma,alpha=sampler,use_alpha=True) 
    optimizer = optim.Adam(model.parameters(), lr=lr,weight_decay=wd)
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps = 0.1*epochs*len(data_loader['train']), num_training_steps = epochs*len(data_loader['train']))
    best_acc = 0
    best_model = None
    best_epoch = 0
    for epoch in range(epochs):
        for phase in ['train','val']:
            running_loss = 0
            running_correct = 0
            for i, (data, target) in enumerate(data_loader[phase]):
                data, target = data.to(device), target.to(device)
                if phase == 'train':
                    model.train()
                    outputs = model(data)
                    loss = criterion(outputs, target)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    scheduler.step()
                else:
                    model.eval()
                    with torch.no_grad():
                        outputs = model(data)
                        loss = criterion(outputs, target)
                running_loss += loss.item()*data.shape[0]
                _,outputs = torch.max(outputs,1)
                running_correct += torch.sum(outputs == target.data)
            epoch_loss = running_loss / len(data_loader[phase].dataset)
            epoch_acc = running_correct.double() / len(data_loader[phase].dataset)
            
            if phase == 'val' and best_acc < epoch_acc:
                best_acc = epoch_acc
                best_epoch = epoch
                best_model = copy.deepcopy(model)
                
            print('[Epoch: %d] [%s] [loss: %.4f] [acc: %.4f]' % (epoch + 1,phase,epoch_loss,epoch_acc))   
    time1 = time.time()
    path = f'./weights/{dataset}_{model_name}_{best_epoch}_{time1}.pth'
    torch.save(best_model.state_dict(),path)
    print('Finished Training')
    return best_epoch, time1
    
def test(device, model, test_loader):
    count = 0
    model.eval()
    y_pred_test = 0
    y_test = 0
    for inputs, labels in test_loader:
        inputs = inputs.to(device)
        outputs = model(inputs)
        outputs = np.argmax(outputs.detach().cpu().numpy(), axis=1)
        if count == 0:
            y_pred_test = outputs
            y_test = labels
            count = 1
        else:
            y_pred_test = np.concatenate((y_pred_test, outputs))
            y_test = np.concatenate((y_test, labels))

    return y_pred_test, y_test
    
def output_metric(tar, pre):
    matrix = confusion_matrix(tar, pre)
    OA, AA_mean, Kappa, AA = cal_results(matrix)
    return OA, AA_mean, Kappa, AA

def cal_results(matrix):
    shape = np.shape(matrix)
    number = 0
    sum = 0
    AA = np.zeros([shape[0]], dtype=np.float64)
    for i in range(shape[0]):
        number += matrix[i, i]
        AA[i] = matrix[i, i] / np.sum(matrix[i, :])
        sum += np.sum(matrix[i, :]) * np.sum(matrix[:, i])
    OA = number / np.sum(matrix)
    AA_mean = np.mean(AA)
    pe = sum / (np.sum(matrix) ** 2)
    Kappa = (OA - pe) / (1 - pe)
    return OA, AA_mean, Kappa, AA

class to_dataset(torch.utils.data.Dataset):
    
    def __init__(self,x,y):

        self.len = x.shape[0]
        self.data = torch.FloatTensor(x)
        self.gt = torch.LongTensor(y)

    def __getitem__(self,index):

        return self.data[index],self.gt[index]
    
    def __len__(self):

        return self.len

class to_dataset_num(torch.utils.data.Dataset):
    
    def __init__(self,x,y):

        self.data = torch.FloatTensor(x)

        self.indices = np.where(y >= 0)

        y = y[y >= 0]

        self.gt = torch.LongTensor(y)

        self.len = y.shape[0]

        print(self.len)

    def __getitem__(self,index):

        return self.data[index],self.gt[index]
    
    def __len__(self):

        return self.len

dataset_name = config['dataset']
model_name = config['model']
train_ratio = float(config['train_ratio'])
patch = config['patch_size']
data,gt = load_data(dataset_name)
batch_size=config['batch_size']

OA_list = []
AA_list = []
Kappa_list = []
EACH_AA = []
Train_time = []
Test_Time = []


def run():


    all_data,all_gt = create_patch(data,gt,patch_size = patch)
    band = all_data.shape[-1]
    classes = int(np.max(np.unique(all_gt)))
    x_train,y_train,x_val,y_val,x_test,y_test,sampler = split_dataset_bynum(all_data,all_gt,train_ratio=train_ratio)
    train_ds = to_dataset_num(x_train,y_train)
    val_ds = to_dataset_num(x_val,y_val)
    test_ds = to_dataset_num(x_test,y_test)
    all_ds = to_dataset_num(all_data,all_gt)
    train_loader = torch.utils.data.DataLoader(train_ds,
                                            batch_size=batch_size,
                                            shuffle=True,
                                            num_workers=0)
                                            
    val_loader = torch.utils.data.DataLoader(val_ds,
                                            batch_size=batch_size,
                                            shuffle=False,
                                            num_workers=0)
    data_loader = {'train':train_loader,
                'val':val_loader}

    test_loader = torch.utils.data.DataLoader(test_ds,
                                            batch_size=batch_size,
                                            shuffle=False,
                                            num_workers=0)
    all_loader = torch.utils.data.DataLoader(all_ds,
                                            batch_size=batch_size,
                                            shuffle=False,
                                            num_workers=0)

    model = DBCTNet(channels=16,
                patch=config['patch_size'],
                bands=band,
                num_class=classes,
                fc_dim=config['fc_dim'],
                heads=config['heads'],
                drop=config['drop'],
                )

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    tic1 = time.time()
    best_epoch,time1 = train(model,classes,config,device,sampler,data_loader)
    tic2 = time.time()
    path = f'./weights/{dataset_name}_{model_name}_{best_epoch}_{time1}.pth'
    model.load_state_dict(torch.load(path))
    model = model.to(device)
    toc1 = time.time()
    p_t,t_t=test(device,model,test_loader)
    toc2 = time.time()
    OA, AA_mean, Kappa, AA=output_metric(t_t, p_t)
    OA_list.append(OA)
    AA_list.append(AA_mean)
    Kappa_list.append(Kappa)
    EACH_AA.append(AA)
    Train_time.append(tic2-tic1)
    Test_Time.append(toc2-toc1)
    # print('Test \n','OA: ',OA,'AA_mean: ',AA_mean,' Kappa: ',Kappa,'AA: ',AA)
    
for i in range(args.n_run):
    run()
print("OA")
print(np.mean(OA_list))
print(np.std(OA_list,ddof=1))
print("AA")
print(np.mean(AA_list))
print("EACH AA")
print(np.mean(EACH_AA,axis=0))
print("Kappa")
print(np.mean(Kappa_list))
print("tr")
print(np.mean(Train_time))
print("te")
print(np.mean(Test_Time))
