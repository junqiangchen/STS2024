import torch
import os
from model import *
from dataprocess.utils import file_name_path, RemoveSmallConnectedCompont2d
import cv2
import pandas as pd

# Use CUDA
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
use_cuda = torch.cuda.is_available()

imaget2_pre = 'image.nii.gz'
mask_pre = '_dsegpd.nii.gz'


def inferenceBinaryVnet2dvalid():
    Vnet2d = BinaryVNet2dModel(image_height=640, image_width=640, image_channel=1, numclass=1,
                               batch_size=4, loss_name='BinaryCrossEntropyDiceLoss', accum_gradient_iter=4,
                               use_cuda=use_cuda, inference=True, model_path=r'log/dicebce2d\BinaryVNet2dModel.pth')
    csv_data2 = pd.read_csv(r'dataprocess/data/valid2d.csv', header=None)
    valimages = csv_data2.iloc[:, 0].values
    vallabels = csv_data2.iloc[:, 1].values
    outmaskpath = r'F:\MedicalData\2023STS2D\validpd'
    if not os.path.exists(outmaskpath):
        os.makedirs(outmaskpath)
    for subsetindex in range(len(valimages)):
        imagedatadir = valimages[subsetindex]
        image = cv2.imread(imagedatadir, 0)
        mask = cv2.imread(vallabels[subsetindex], 0)
        mask_pd = Vnet2d.inference(image)
        mask_pd = RemoveSmallConnectedCompont2d(mask_pd)
        cv2.imwrite(outmaskpath + '/' + str(subsetindex) + "_pd.png", mask_pd * 255)
        cv2.imwrite(outmaskpath + '/' + str(subsetindex) + "_gt.png", mask)
        cv2.imwrite(outmaskpath + '/' + str(subsetindex) + ".png", image)
    Vnet2d.clear_GPU_cache()


def inferenceBinaryVnet2dtest():
    Vnet2d = BinaryVNet2dModel(image_height=640, image_width=640, image_channel=1, numclass=1,
                               batch_size=4, loss_name='BinaryCrossEntropyDiceLoss', accum_gradient_iter=4,
                               use_cuda=use_cuda, inference=True, model_path=r'log/dicebce2d\BinaryVNet2dModel.pth')
    datapath = r"F:\MedicalData\2024Semi-TeethSeg\2D XRAY\Validation-Public"
    outmaskpath = r'F:\MedicalData\2024Semi-TeethSeg\2D XRAY\Validation-Public-Mask'
    if not os.path.exists(outmaskpath):
        os.makedirs(outmaskpath)
    filenames = file_name_path(datapath, False, True)
    for subsetindex in range(len(filenames)):
        imagedatadir = datapath + '/' + filenames[subsetindex]
        image = cv2.imread(imagedatadir, 0)
        mask_pd = Vnet2d.inference(image)
        mask_pd = RemoveSmallConnectedCompont2d(mask_pd)
        cv2.imwrite(outmaskpath + '/' + filenames[subsetindex], mask_pd * 255)
    Vnet2d.clear_GPU_cache()


if __name__ == '__main__':
    # inferenceBinaryVnet2dvalid()
    inferenceBinaryVnet2dtest()
