import pandas as pd
import torch
import os
from model import *
import numpy as np

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
# Use CUDA
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
use_cuda = torch.cuda.is_available()


def trainMutilVNet2d():
    # # Read  data set (Train data from CSV file)
    csvdata = pd.read_csv('dataprocess\\data\\train2d.csv', header=None)
    maskdatasource = csvdata.iloc[:, 1].values
    imagedatasource = csvdata.iloc[:, 0].values
    csvdataaug = pd.read_csv('dataprocess\\data\\unlabel_train2d.csv', header=None)
    maskdataaug = csvdataaug.iloc[:, 1].values
    imagedataaug = csvdataaug.iloc[:, 0].values
    trainimages = np.concatenate((imagedatasource, imagedataaug), axis=0)
    trainlabels = np.concatenate((maskdatasource, maskdataaug), axis=0)

    csv_data2 = pd.read_csv(r'dataprocess/data/valid2d.csv', header=None)
    valimages = csv_data2.iloc[:, 0].values
    vallabels = csv_data2.iloc[:, 1].values

    Vnet2d = MutilVNet2dModel(image_height=1024, image_width=1024, image_channel=1, numclass=53,
                              batch_size=6, loss_name='MutilCrossEntropyDiceLoss', accum_gradient_iter=1)
    Vnet2d.trainprocess(trainimages, trainlabels, valimages, vallabels, model_dir='log/dicece2d', epochs=500)
    Vnet2d.clear_GPU_cache()


def trainMutilVNet3d():
    # # Read  data set (Train data from CSV file)
    csvdata = pd.read_csv('dataprocess\\data\\train3d.csv', header=None)
    maskdatasource = csvdata.iloc[:, 1].values
    imagedatasource = csvdata.iloc[:, 0].values
    csvdataaug = pd.read_csv('dataprocess\\data\\unlabel_train3d.csv', header=None)
    maskdataaug = csvdataaug.iloc[:, 1].values
    imagedataaug = csvdataaug.iloc[:, 0].values
    imagedata = np.concatenate((imagedatasource, imagedataaug), axis=0)
    maskdata = np.concatenate((maskdatasource, maskdataaug), axis=0)
    # shuffle imagedata and maskdata together
    perm = np.arange(len(imagedata))
    np.random.shuffle(perm)
    trainimages = imagedata[perm]
    trainlabels = maskdata[perm]

    csv_data2 = pd.read_csv(r'dataprocess/data/valid3d.csv', header=None)
    valimages = csv_data2.iloc[:, 0].values
    vallabels = csv_data2.iloc[:, 1].values

    Vnet3d = MutilVNet3dModel(image_depth=160, image_height=256, image_width=256, image_channel=1, numclass=35,
                              batch_size=1, loss_name='MutilCrossEntropyDiceLoss', accum_gradient_iter=4)
    Vnet3d.trainprocess(trainimages, trainlabels, valimages, vallabels, model_dir='log/dicece3d', epochs=300,
                        showwind=[10, 16])
    Vnet3d.clear_GPU_cache()


if __name__ == '__main__':
    trainMutilVNet2d()
    trainMutilVNet3d()
