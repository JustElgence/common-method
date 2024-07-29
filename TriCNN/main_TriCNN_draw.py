'''
Code implementation of "Tri-CNN: A Three Branch Model for Hyperspectral Image Classification"

Mohammed Q. Alkhatib (mqalkhatib@ieee.org)    
'''

from tensorflow import keras
from sklearn.metrics import confusion_matrix, accuracy_score, cohen_kappa_score
import numpy as np
import scipy
from utils import *
from model_TriCNN import Tri_CNN
import time
import visdom
import seaborn as sns

## GLOBAL VARIABLES
dataset = 'PC'
windowSize = 9
PCA_comp = 45

KAPPA = []
OA = []
AA = []
TRAINING_TIME = []
TESTING_TIME = []
from tensorflow.keras.callbacks import EarlyStopping
early_stopper = EarlyStopping(monitor='accuracy', 
                            patience=10,
                            restore_best_weights=True
                            )


def convert_to_color_(arr_2d, palette=None):

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

viz = visdom.Visdom(env = "Draw")

if not viz.check_connection:
	print("Visdom is not connected. Did you run 'python -m visdom.server' ?")
    
def display_predictions(pred, vis, gt=None, caption=""):
    if gt is None:

        vis.images([np.transpose(pred, (2, 0, 1))],
                    opts={'caption': caption})
    else:

        vis.images([np.transpose(pred, (2, 0, 1)),
                    np.transpose(gt, (2, 0, 1))],
                    nrow=2,
                    opts={'caption': caption})

EACH_AA = []
for run in range(0, 1):
    print("Epoch " + str(run))
    X, y = loadData(dataset)
    
    X.shape, y.shape

    # Apply PCA for dimensionality reduction
    X,pca = applyPCA(X,PCA_comp)


    X, y = createImageCubes(X, y, windowSize=windowSize)


    Xtrain, Xtest, ytrain, ytest = splitTrainTestSet(X, y, 0.99)

    ytrain = keras.utils.to_categorical(ytrain)

    Xtrain = np.expand_dims(Xtrain, axis=4)
    Xtest = np.expand_dims(Xtest, axis=4)




    # Early stopper: if the model stopped improving for 10 consecutive epochs, 
    # stop the training

    tic1 = time.time()
    model = Tri_CNN(Xtrain, num_classes(dataset))
    model.summary()
    
    history = model.fit(Xtrain, ytrain,
                        batch_size = 4, 
                        verbose=1, 
                        epochs=100, 
                        shuffle=True, 
                        callbacks = [early_stopper])
    del history
    toc1 = time.time()
    tic2 = time.time()
    Y_pred_test = model.predict(Xtest)
    y_pred_test = np.argmax(Y_pred_test, axis=1)
        
    toc2 = time.time()
    kappa = cohen_kappa_score(ytest,  y_pred_test)
    oa = accuracy_score(ytest, y_pred_test)
    confusion = confusion_matrix(ytest, y_pred_test)
    print(confusion)
    each_acc, aa = AA_andEachClassAccuracy(confusion)
    EACH_AA.append(each_acc)
    KAPPA.append(kappa)
    OA.append(oa)
    AA.append(aa)
    TRAINING_TIME.append(toc1 - tic1)
    TESTING_TIME.append(toc2 - tic2)
    print(each_acc)
    del X, y



    X, y = loadData(dataset)
    height = y.shape[0]
    width = y.shape[1]
    X,pca = applyPCA(X, numComponents=PCA_comp)
    X = padWithZeros(X, windowSize//2)

    # calculate the predicted image, this is a pixel wise operation, will take long time
    outputs = np.zeros((height,width))
    for i in range(height):
        for j in range(width):
            target = int(y[i,j])
            if i%25 == 0 and j%25 ==0: 
                print("i = " + str(i) + ", j = " + str(j))
            if target == 0 :
                continue
            else :
                image_patch=Patch(X,i,j, windowSize)
                X_test_image = image_patch.reshape(1, image_patch.shape[0],
                                                image_patch.shape[1], 
                                                image_patch.shape[2], 1).astype('float32')                                   
                prediction = (model.predict(X_test_image))
                prediction = np.argmax(prediction, axis=1)
                outputs[i][j] = prediction + 1
    
    display_predictions(convert_to_color(outputs, 10, palette = None), viz, caption="prediction")
mean_AA = np.mean(EACH_AA, axis=0)
print("EACH AA:")
# for i in range(num_classes(dataset)):
#     print(mean_AA[i])
KAPPA1 = np.array(KAPPA)
OA1 = np.array(OA)
AA1 = np.array(AA)
TRAINING_TIME1 = np.array(TRAINING_TIME)
TESTING_TIME1 = np.array(TESTING_TIME)   
print(OA)
print("OA is {}, AA is {}, kappa is {}, Tr_time is {}, Te_time is {}, Std OA is {}".format(np.mean(OA1), \
    np.mean(AA1), np.mean(KAPPA1), np.mean(TRAINING_TIME1), np.mean(TESTING_TIME1), np.std(OA, ddof=1)))


