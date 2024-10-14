import numpy as np
import torch
import os
from model import *
from dataprocess.utils import file_name_path, RemoveSmallConnectedCompont2d
import cv2
import pandas as pd
from dataprocess.STS2ddataprocess import convertmasktojson
import json

# Use CUDA
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
use_cuda = torch.cuda.is_available()

imaget2_pre = 'image.nii.gz'
mask_pre = '_dsegpd.nii.gz'


def inferenceMutilVnet2dvalid():
    Vnet2d = MutilVNet2dModel(image_height=1024, image_width=1024, image_channel=1, numclass=53,
                              batch_size=4, loss_name='MutilCrossEntropyDiceLoss', accum_gradient_iter=1,
                              use_cuda=use_cuda, inference=True, model_path=r'log/dicece2d\MutilVNet2dModel.pth')
    csv_data2 = pd.read_csv(r'dataprocess/data/valid2d.csv', header=None)
    valimages = csv_data2.iloc[:, 0].values
    vallabels = csv_data2.iloc[:, 1].values
    outmaskpath = r'F:\MedicalData\2024Semi-TeethSeg\XRAY_validpd'
    if not os.path.exists(outmaskpath):
        os.makedirs(outmaskpath)
    for subsetindex in range(len(valimages)):
        imagedatadir = valimages[subsetindex]
        image = cv2.imread(imagedatadir, 0)
        mask = cv2.imread(vallabels[subsetindex], 0)
        mask_pd = Vnet2d.inference(image)
        mask_pd_binary = mask_pd.copy()
        mask_pd_binary[mask_pd_binary != 0] = 1
        mask_pd_binary = RemoveSmallConnectedCompont2d(mask_pd_binary)
        mask_pd[mask_pd_binary == 0] = 0
        cv2.imwrite(outmaskpath + '/' + str(subsetindex) + "_pd.png", mask_pd * 4.5)
        cv2.imwrite(outmaskpath + '/' + str(subsetindex) + "_gt.png", mask * 4.5)
        cv2.imwrite(outmaskpath + '/' + str(subsetindex) + ".png", image)
    Vnet2d.clear_GPU_cache()


def inferenceMutilVnet2dtest():
    """
    使用STS2023年的数据先分割牙齿区域，然后在牙齿区域中分割牙齿细分割。
    """
    Vnet2d1 = BinaryVNet2dModel(image_height=640, image_width=640, image_channel=1, numclass=1,
                                batch_size=4, loss_name='BinaryCrossEntropyDiceLoss', accum_gradient_iter=4,
                                use_cuda=use_cuda, inference=True, model_path=r'log/dicebce2d\BinaryVNet2dModel.pth')
    Vnet2d2 = MutilVNet2dModel(image_height=1024, image_width=1024, image_channel=1, numclass=53,
                               batch_size=4, loss_name='MutilCrossEntropyDiceLoss', accum_gradient_iter=1,
                               use_cuda=use_cuda, inference=True, model_path=r'log/dicece2d\MutilVNet2dModel.pth')
    datapath = r"F:\MedicalData\2024Semi-TeethSeg\2D XRAY\Train-Unlabeled"
    outmaskpath = r'F:\MedicalData\2024Semi-TeethSeg\2D XRAY\Train-Unlabeled-Mask'
    if not os.path.exists(outmaskpath):
        os.makedirs(outmaskpath)
    filenames = file_name_path(datapath, False, True)
    for subsetindex in range(len(filenames)):
        imagedatadir = datapath + '/' + filenames[subsetindex]
        image = cv2.imread(imagedatadir, 0)
        mask_pd1 = Vnet2d1.inference(image)
        mask_pd1 = RemoveSmallConnectedCompont2d(mask_pd1)
        dilated_mask_pd1 = cv2.dilate(mask_pd1, np.ones((5, 5), np.uint8), iterations=1)
        mask_pd2 = Vnet2d2.inference(image)
        mask_pd2[dilated_mask_pd1 == 0] = 0
        cv2.imwrite(outmaskpath + '/' + filenames[subsetindex][:-4] + "mutil.jpg", mask_pd2 * 4.5)
        cv2.imwrite(outmaskpath + '/' + filenames[subsetindex][:-4] + "binary.jpg", mask_pd1 * 255)
    Vnet2d1.clear_GPU_cache()
    Vnet2d2.clear_GPU_cache()


def inferenceMutilVnet2dtestjson():
    """
        使用STS2023年的数据先分割牙齿区域，然后在牙齿区域中分割牙齿细分割。
        """
    Vnet2d1 = BinaryVNet2dModel(image_height=640, image_width=640, image_channel=1, numclass=1,
                                batch_size=4, loss_name='BinaryCrossEntropyDiceLoss', accum_gradient_iter=4,
                                use_cuda=use_cuda, inference=True, model_path=r'log/dicebce2d\BinaryVNet2dModel.pth')
    Vnet2d2 = MutilVNet2dModel(image_height=1024, image_width=1024, image_channel=1, numclass=53,
                               batch_size=4, loss_name='MutilCrossEntropyDiceLoss', accum_gradient_iter=1,
                               use_cuda=use_cuda, inference=True, model_path=r'log/dicece2d\MutilVNet2dModel.pth')
    datapath = r"F:\MedicalData\2024Semi-TeethSeg\2D XRAY\Validation-Public"
    outmaskpath = r'F:\MedicalData\2024Semi-TeethSeg\2D XRAY\Validation-Public-json'
    if not os.path.exists(outmaskpath):
        os.makedirs(outmaskpath)
    filenames = file_name_path(datapath, False, True)
    for subsetindex in range(len(filenames)):
        imagedatadir = datapath + '/' + filenames[subsetindex]
        image = cv2.imread(imagedatadir, 0)
        mask_pd1 = Vnet2d1.inference(image)
        mask_pd1 = RemoveSmallConnectedCompont2d(mask_pd1)
        dilated_mask_pd1 = cv2.dilate(mask_pd1, np.ones((5, 5), np.uint8), iterations=1)
        mask_pd2 = Vnet2d2.inference(image)
        mask_pd2[dilated_mask_pd1 == 0] = 0
        json_data = convertmasktojson(mask_pd2)
        json_file = outmaskpath + '/' + "Validation_" + filenames[subsetindex][-7:-4] + "_Mask.json"
        with open(json_file, 'w') as json_file:
            json.dump(json_data, json_file, indent=4)
    Vnet2d1.clear_GPU_cache()
    Vnet2d2.clear_GPU_cache()


def inferenceMutilVnet2dtestv2():
    """
         直接对牙齿进行细分割，然后根据细分割的结果去除小目标后得到分割结果，
            效果比用STS2023年的数据先分割牙齿区域，然后在牙齿区域中分割牙齿细分割。
        """
    Vnet2d2 = MutilVNet2dModel(image_height=1024, image_width=1024, image_channel=1, numclass=53,
                               batch_size=4, loss_name='MutilCrossEntropyDiceLoss', accum_gradient_iter=1,
                               use_cuda=use_cuda, inference=True, model_path=r'log/dicece2d\MutilVNet2dModel.pth')
    datapath = r"F:\MedicalData\2024Semi-TeethSeg\2D XRAY\Train-Unlabeled"
    outmaskpath = r'F:\MedicalData\2024Semi-TeethSeg\2D XRAY\Train-Unlabeled-Mask'
    if not os.path.exists(outmaskpath):
        os.makedirs(outmaskpath)
    filenames = file_name_path(datapath, False, True)
    for subsetindex in range(len(filenames)):
        imagedatadir = datapath + '/' + filenames[subsetindex]
        image = cv2.imread(imagedatadir, 0)
        mask_pd = Vnet2d2.inference(image)
        mask_pd_binary = mask_pd.copy()
        mask_pd_binary[mask_pd_binary != 0] = 1
        mask_pd_binary = RemoveSmallConnectedCompont2d(mask_pd_binary)
        mask_pd[mask_pd_binary == 0] = 0
        cv2.imwrite(outmaskpath + '/' + filenames[subsetindex][:-4] + "mutil.bmp", mask_pd)
    Vnet2d2.clear_GPU_cache()


def inferenceMutilVnet2dtestjsonv2():
    """
            直接对牙齿进行细分割，然后根据细分割的结果去除小目标后得到分割结果，
            效果比用STS2023年的数据先分割牙齿区域，然后在牙齿区域中分割牙齿细分割。
            """
    Vnet2d2 = MutilVNet2dModel(image_height=1024, image_width=1024, image_channel=1, numclass=53,
                               batch_size=4, loss_name='MutilCrossEntropyDiceLoss', accum_gradient_iter=1,
                               use_cuda=use_cuda, inference=True, model_path=r'log/dicece2d\MutilVNet2dModel.pth')
    datapath = r"F:\MedicalData\2024Semi-TeethSeg\2D XRAY\Validation-Public"
    outmaskpath = r'F:\MedicalData\2024Semi-TeethSeg\2D XRAY\Validation-Public-json'
    if not os.path.exists(outmaskpath):
        os.makedirs(outmaskpath)
    filenames = file_name_path(datapath, False, True)
    for subsetindex in range(len(filenames)):
        imagedatadir = datapath + '/' + filenames[subsetindex]
        image = cv2.imread(imagedatadir, 0)
        mask_pd = Vnet2d2.inference(image)
        mask_pd_binary = mask_pd.copy()
        mask_pd_binary[mask_pd_binary != 0] = 1
        mask_pd_binary = RemoveSmallConnectedCompont2d(mask_pd_binary)
        mask_pd[mask_pd_binary == 0] = 0
        json_data = convertmasktojson(mask_pd)
        json_file = outmaskpath + '/' + "Validation_" + filenames[subsetindex][-7:-4] + "_Mask.json"
        with open(json_file, 'w') as json_file:
            json.dump(json_data, json_file, indent=4)
    Vnet2d2.clear_GPU_cache()


if __name__ == '__main__':
    # inferenceMutilVnet2dvalid()
    # inferenceMutilVnet2dtest()
    # inferenceMutilVnet2dtestjson()
    # inferenceMutilVnet2dtestv2()
    inferenceMutilVnet2dtestjsonv2()
