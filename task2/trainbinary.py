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


def trainBinaryVNet2d():
    # # Read  data set (Train data from CSV file)
    csvdata = pd.read_csv('dataprocess\\data\\train2d.csv', header=None)
    maskdata = csvdata.iloc[:, 1].values
    imagedata = csvdata.iloc[:, 0].values
    # shuffle imagedata and maskdata together
    perm = np.arange(len(imagedata))
    np.random.shuffle(perm)
    trainimages = imagedata[perm]
    trainlabels = maskdata[perm]

    csv_data2 = pd.read_csv(r'dataprocess/data/valid2d.csv', header=None)
    valimages = csv_data2.iloc[:, 0].values
    vallabels = csv_data2.iloc[:, 1].values

    Vnet2d = BinaryVNet2dModel(image_height=640, image_width=640, image_channel=1, numclass=1,
                               batch_size=8, loss_name='BinaryCrossEntropyDiceLoss', accum_gradient_iter=1)
    Vnet2d.trainprocess(trainimages, trainlabels, valimages, vallabels, model_dir='log/dicebce2d', epochs=300)
    Vnet2d.clear_GPU_cache()


def trainBinaryVNet3d():
    # # Read  data set (Train data from CSV file)
    csvdata = pd.read_csv('dataprocess\\data\\train3d.csv', header=None)
    maskdatasource = csvdata.iloc[:, 1].values
    imagedatasource = csvdata.iloc[:, 0].values
    csvdataaug = pd.read_csv('dataprocess\\data\\augtrain3d.csv', header=None)
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

    Vnet3d = BinaryVNet3dModel(image_depth=256, image_height=256, image_width=256, image_channel=1, numclass=1,
                               batch_size=1, loss_name='BinaryCrossEntropyDiceLoss', accum_gradient_iter=4)
    Vnet3d.trainprocess(trainimages, trainlabels, valimages, vallabels, model_dir='log/dicebce3d', epochs=300,
                        showwind=[16, 16])
    Vnet3d.clear_GPU_cache()


if __name__ == '__main__':
    trainBinaryVNet2d()
    trainBinaryVNet3d()
