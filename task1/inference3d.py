import torch
import os
from model import *
from dataprocess.utils import file_name_path, GetLargestConnectedCompontBoundingbox, RemoveSmallConnectedCompont, \
    GetLargestConnectedCompont, MorphologicalOperation
from dataprocess.STSdataprocess import getbrain
import SimpleITK as sitk
import numpy as np

# Use CUDA
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
use_cuda = torch.cuda.is_available()

imaget2_pre = 'image.nii.gz'
mask_pre = '_dsegpd.nii.gz'


def inferenceBinaryVnet3dvalid():
    Vnet3d = BinaryVNet3dModel(image_depth=256, image_height=256, image_width=256, image_channel=1, numclass=1,
                               batch_size=1, loss_name='BinaryCrossEntropyDiceLoss', accum_gradient_iter=4,
                               use_cuda=use_cuda, inference=True, model_path=r'log/dicebce3d\BinaryVNet3d.pth')
    datapath = r"F:\MedicalData\2023STS3D\roiprocess\validation"
    outmaskpath = r'F:\MedicalData\2023STS3D\roiprocess\validation\pd'
    if not os.path.exists(outmaskpath):
        os.makedirs(outmaskpath)
    image_path_list = file_name_path(datapath, False, True)
    for i in range(len(image_path_list)):
        if imaget2_pre in image_path_list[i]:
            imaget2pathname = datapath + "/" + image_path_list[i]
            sitk_imaget2 = sitk.ReadImage(imaget2pathname)
            sitk_mask = Vnet3d.inference(sitk_imaget2, newSize=(256, 256, 256))
            sitk_mask = RemoveSmallConnectedCompont(sitk_mask, 0.1)
            maskpathname = outmaskpath + "/" + image_path_list[i]
            sitk.WriteImage(sitk_mask, maskpathname)


def inferenceBinaryVnet3dtestchusai():
    Vnet3d = BinaryVNet3dModel(image_depth=256, image_height=256, image_width=256, image_channel=1, numclass=1,
                               batch_size=1, loss_name='BinaryCrossEntropyDiceLoss', accum_gradient_iter=4,
                               use_cuda=use_cuda, inference=True, model_path=r'log/dicebce3d\BinaryVNet3d.pth')
    datapath = r"F:\MedicalData\2023STS3D\download\unlabelled_image_1"
    outmaskpath = r'F:\MedicalData\2023STS3D\download\unlabelled_mask_1'
    if not os.path.exists(outmaskpath):
        os.makedirs(outmaskpath)
    filenames = file_name_path(datapath, False, True)
    for subsetindex in range(len(filenames)):
        imagedatadir = datapath + '/' + filenames[subsetindex]
        sitk_imaget2 = sitk.ReadImage(imagedatadir)
        # brain region segmentation
        brainmasksitk = getbrain(sitk_imaget2)
        bodyboundingbox = GetLargestConnectedCompontBoundingbox(brainmasksitk)
        x1, y1, z1, x2, y2, z2 = bodyboundingbox[0], bodyboundingbox[1], \
                                 bodyboundingbox[2], bodyboundingbox[0] + bodyboundingbox[3], \
                                 bodyboundingbox[1] + bodyboundingbox[4], bodyboundingbox[2] + \
                                 bodyboundingbox[5]
        src_array_t2 = sitk.GetArrayFromImage(sitk_imaget2)
        roi_src_array_t2 = src_array_t2[z1:z2, y1:y2, x1:x2]
        roi_src_t2 = sitk.GetImageFromArray(roi_src_array_t2)
        roi_src_t2.SetSpacing(sitk_imaget2.GetSpacing())
        roi_src_t2.SetDirection(sitk_imaget2.GetDirection())
        roi_src_t2.SetOrigin(sitk_imaget2.GetOrigin())

        sitk_mask_roi = Vnet3d.inference(roi_src_t2, newSize=(256, 256, 256))

        sitk_mask_roi = RemoveSmallConnectedCompont(sitk_mask_roi, 0.1)

        final_mask_array = np.zeros_like(src_array_t2)
        array_mask_roi = sitk.GetArrayFromImage(sitk_mask_roi)
        final_mask_array[z1:z2, y1:y2, x1:x2] = array_mask_roi.copy()
        final_mask_sitk = sitk.GetImageFromArray(final_mask_array.astype('uint8'))
        final_mask_sitk.SetSpacing(sitk_imaget2.GetSpacing())
        final_mask_sitk.SetDirection(sitk_imaget2.GetDirection())
        final_mask_sitk.SetOrigin(sitk_imaget2.GetOrigin())

        maskpathname = outmaskpath + "/" + filenames[subsetindex]
        sitk.WriteImage(final_mask_sitk, maskpathname)


def inferenceBinaryVnet3dtestfusai():
    Vnet3d = BinaryVNet3dModel(image_depth=256, image_height=256, image_width=256, image_channel=1, numclass=1,
                               batch_size=1, loss_name='BinaryCrossEntropyDiceLoss', accum_gradient_iter=4,
                               use_cuda=use_cuda, inference=True, model_path=r'log/dicebce3d\BinaryVNet3d.pth')
    datapath = r"F:\MedicalData\2023STS3D\download\fusai\test"
    outmaskpath = r'F:\MedicalData\2023STS3D\download\fusai\testmask'
    if not os.path.exists(outmaskpath):
        os.makedirs(outmaskpath)
    filenames = file_name_path(datapath, False, True)
    for subsetindex in range(len(filenames)):
        imagedatadir = datapath + '/' + filenames[subsetindex]
        sitk_imaget2 = sitk.ReadImage(imagedatadir)
        # brain region segmentation
        brainmasksitk = getbrain(sitk_imaget2)
        bodyboundingbox = GetLargestConnectedCompontBoundingbox(brainmasksitk)
        x1, y1, z1, x2, y2, z2 = bodyboundingbox[0], bodyboundingbox[1], \
                                 bodyboundingbox[2], bodyboundingbox[0] + bodyboundingbox[3], \
                                 bodyboundingbox[1] + bodyboundingbox[4], bodyboundingbox[2] + \
                                 bodyboundingbox[5]
        src_array_t2 = sitk.GetArrayFromImage(sitk_imaget2)
        roi_src_array_t2 = src_array_t2[z1:z2, y1:y2, x1:x2]
        roi_src_t2 = sitk.GetImageFromArray(roi_src_array_t2)
        roi_src_t2.SetSpacing(sitk_imaget2.GetSpacing())
        roi_src_t2.SetDirection(sitk_imaget2.GetDirection())
        roi_src_t2.SetOrigin(sitk_imaget2.GetOrigin())

        sitk_mask_roi = Vnet3d.inference(roi_src_t2, newSize=(256, 256, 256))

        sitk_mask_roi = RemoveSmallConnectedCompont(sitk_mask_roi, 0.1)
        array_mask_roi = sitk.GetArrayFromImage(sitk_mask_roi)
        final_mask_array1 = np.zeros_like(src_array_t2)
        final_mask_array1[z1:z2, y1:y2, x1:x2] = array_mask_roi.copy()
        # crop center half part
        z_depth = sitk_imaget2.GetSize()[2]
        z1 = z_depth // 2 - z_depth // 4
        z2 = z_depth // 2 + z_depth // 4
        print((z1, z2))
        final_mask_array = np.zeros_like(src_array_t2)
        final_mask_array[z1:z2, :, :] = final_mask_array1[z1:z2, :, :]
        final_mask_sitk = sitk.GetImageFromArray(final_mask_array.astype('uint8'))
        final_mask_sitk.SetSpacing(sitk_imaget2.GetSpacing())
        final_mask_sitk.SetDirection(sitk_imaget2.GetDirection())
        final_mask_sitk.SetOrigin(sitk_imaget2.GetOrigin())

        maskpathname = outmaskpath + "/" + filenames[subsetindex]
        sitk.WriteImage(final_mask_sitk, maskpathname)


def inferenceBinaryVnet3dtestfusaiv2():
    Vnet3d = BinaryVNet3dModel(image_depth=256, image_height=256, image_width=256, image_channel=1, numclass=1,
                               batch_size=1, loss_name='BinaryCrossEntropyDiceLoss', accum_gradient_iter=4,
                               use_cuda=use_cuda, inference=True, model_path=r'log/dicebce3d\BinaryVNet3d.pth')
    datapath = r"F:\MedicalData\2023STS3D\download\fusai\unlabelled"
    outmaskpath = r'F:\MedicalData\2023STS3D\download\fusai\unlabelledmask'
    if not os.path.exists(outmaskpath):
        os.makedirs(outmaskpath)
    filenames = file_name_path(datapath, False, True)
    for subsetindex in range(len(filenames)):
        imagedatadir = datapath + '/' + filenames[subsetindex]
        sitk_imaget2 = sitk.ReadImage(imagedatadir)
        # brain region segmentation
        brainmasksitk = getbrain(sitk_imaget2)
        bodyboundingbox = GetLargestConnectedCompontBoundingbox(brainmasksitk)
        x1, y1, z1, x2, y2, z2 = bodyboundingbox[0], bodyboundingbox[1], \
                                 bodyboundingbox[2], bodyboundingbox[0] + bodyboundingbox[3], \
                                 bodyboundingbox[1] + bodyboundingbox[4], bodyboundingbox[2] + \
                                 bodyboundingbox[5]
        src_array_t2 = sitk.GetArrayFromImage(sitk_imaget2)
        roi_src_array_t2 = src_array_t2[z1:z2, y1:y2, x1:x2]
        roi_src_t2 = sitk.GetImageFromArray(roi_src_array_t2)
        roi_src_t2.SetSpacing(sitk_imaget2.GetSpacing())
        roi_src_t2.SetDirection(sitk_imaget2.GetDirection())
        roi_src_t2.SetOrigin(sitk_imaget2.GetOrigin())

        sitk_mask_roi = Vnet3d.inference(roi_src_t2, newSize=(256, 256, 256))

        sitk_mask_roi = RemoveSmallConnectedCompont(sitk_mask_roi, 0.1)
        array_mask_roi = sitk.GetArrayFromImage(sitk_mask_roi)
        final_mask_array = np.zeros_like(src_array_t2)
        final_mask_array[z1:z2, y1:y2, x1:x2] = array_mask_roi.copy()
        final_mask_sitk = sitk.GetImageFromArray(final_mask_array.astype('uint8'))
        final_mask_sitk.SetSpacing(sitk_imaget2.GetSpacing())
        final_mask_sitk.SetDirection(sitk_imaget2.GetDirection())
        final_mask_sitk.SetOrigin(sitk_imaget2.GetOrigin())

        maskpathname = outmaskpath + "/" + filenames[subsetindex]
        sitk.WriteImage(final_mask_sitk, maskpathname)


if __name__ == '__main__':
    # inferenceBinaryVnet3dvalid()
    inferenceBinaryVnet3dtestchusai()
    # inferenceBinaryVnet3dtestfusai()
    inferenceBinaryVnet3dtestfusaiv2()
