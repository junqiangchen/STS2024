import torch
import os
from model import *
from dataprocess.utils import file_name_path, GetLargestConnectedCompontBoundingbox, RemoveSmallConnectedCompont, \
    GetLargestConnectedCompont, MorphologicalOperation, getRangImageRange
from dataprocess.STSdataprocess import getbrain
import SimpleITK as sitk
import numpy as np

# Use CUDA
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
use_cuda = torch.cuda.is_available()

imaget2_pre = 'image.nii.gz'
mask_pre = '_label.nii.gz'


def inferenceMutilVnet3dvalid():
    Vnet3d = MutilVNet3dModel(image_depth=160, image_height=256, image_width=256, image_channel=1, numclass=35,
                              batch_size=1, loss_name='MutilCrossEntropyDiceLoss', accum_gradient_iter=4,
                              use_cuda=use_cuda, inference=True, model_path=r'log/dicece3d/MutilVNet3dModel.pth')
    datapath = r"F:\MedicalData\2024Semi-TeethSeg\roiprocessCBCT\validation"
    outmaskpath = r'F:\MedicalData\2024Semi-TeethSeg\roiprocessCBCT\validation\pd'
    if not os.path.exists(outmaskpath):
        os.makedirs(outmaskpath)
    image_path_list = file_name_path(datapath, False, True)
    for i in range(len(image_path_list)):
        if imaget2_pre in image_path_list[i]:
            imaget2pathname = datapath + "/" + image_path_list[i]
            sitk_imaget2 = sitk.ReadImage(imaget2pathname)
            sitk_mask = Vnet3d.inference(sitk_imaget2, newSize=(256, 256, 160))
            final_mask_array2 = sitk.GetArrayFromImage(sitk_mask)
            organmasksitk = RemoveSmallConnectedCompont(sitk_mask, 0.1)
            final_mask_array2[sitk.GetArrayFromImage(organmasksitk) == 0] = 0
            final_mask_sitk2 = sitk.GetImageFromArray(final_mask_array2.astype('uint8'))
            final_mask_sitk2.SetSpacing(sitk_imaget2.GetSpacing())
            final_mask_sitk2.SetDirection(sitk_imaget2.GetDirection())
            final_mask_sitk2.SetOrigin(sitk_imaget2.GetOrigin())
            maskpathname = outmaskpath + "/" + image_path_list[i][:-len(imaget2_pre)] + mask_pre
            sitk.WriteImage(final_mask_sitk2, maskpathname)


def inferenceMutilVnet3dtest():
    Vnet3d_binary = BinaryVNet3dModel(image_depth=256, image_height=256, image_width=256, image_channel=1, numclass=1,
                                      batch_size=1, loss_name='BinaryCrossEntropyDiceLoss', accum_gradient_iter=4,
                                      use_cuda=use_cuda, inference=True, model_path=r'log/dicebce3d\BinaryVNet3d.pth')

    Vnet3d_mutil = MutilVNet3dModel(image_depth=160, image_height=256, image_width=256, image_channel=1, numclass=35,
                                    batch_size=1, loss_name='MutilCrossEntropyDiceLoss', accum_gradient_iter=4,
                                    use_cuda=use_cuda, inference=True, model_path=r'log/dicece3d/MutilVNet3dModel.pth')
    datapath = r"F:\MedicalData\2024Semi-TeethSeg\3D CBCT\Train-Unlabeled-update"
    outmaskpath = r'F:\MedicalData\2024Semi-TeethSeg\3D CBCT\Train-Unlabeled-update-Mask'
    imagetype = ".nii.gz"
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
        # binary segmentation
        sitk_mask_roi = Vnet3d_binary.inference(roi_src_t2, newSize=(256, 256, 256))
        sitk_mask_roi = RemoveSmallConnectedCompont(sitk_mask_roi, 0.1)
        array_mask_roi = sitk.GetArrayFromImage(sitk_mask_roi)
        final_mask_array1 = np.zeros_like(src_array_t2)
        final_mask_array1[z1:z2, y1:y2, x1:x2] = array_mask_roi.copy()
        final_mask_sitk1 = sitk.GetImageFromArray(final_mask_array1.astype('uint8'))
        final_mask_sitk1.SetSpacing(sitk_imaget2.GetSpacing())
        final_mask_sitk1.SetDirection(sitk_imaget2.GetDirection())
        final_mask_sitk1.SetOrigin(sitk_imaget2.GetOrigin())
        # body region segmentation
        organmasksitk = MorphologicalOperation(final_mask_sitk1, 10, "dilate")
        organmasksitk = GetLargestConnectedCompont(organmasksitk)
        organmaskarray = sitk.GetArrayFromImage(organmasksitk)
        z1, z2 = getRangImageRange(organmaskarray, 0)
        y1, y2 = getRangImageRange(organmaskarray, 1)
        x1, x2 = getRangImageRange(organmaskarray, 2)
        src_array_t2 = sitk.GetArrayFromImage(sitk_imaget2)
        roi_src_array_t2 = src_array_t2[z1:z2, y1:y2, x1:x2]
        roi_src_t2 = sitk.GetImageFromArray(roi_src_array_t2)
        roi_src_t2.SetSpacing(sitk_imaget2.GetSpacing())
        roi_src_t2.SetDirection(sitk_imaget2.GetDirection())
        roi_src_t2.SetOrigin(sitk_imaget2.GetOrigin())
        sitk_mask_roi2 = Vnet3d_mutil.inference(roi_src_t2, newSize=(256, 256, 160))
        array_mask_roi2 = sitk.GetArrayFromImage(sitk_mask_roi2)
        final_mask_array2 = np.zeros_like(src_array_t2)
        final_mask_array2[z1:z2, y1:y2, x1:x2] = array_mask_roi2.copy()
        organmasksitk = MorphologicalOperation(final_mask_sitk1, 3, "dilate")
        final_mask_array2[sitk.GetArrayFromImage(organmasksitk) == 0] = 0
        final_mask_sitk2 = sitk.GetImageFromArray(final_mask_array2.astype('uint8'))
        final_mask_sitk2.SetSpacing(sitk_imaget2.GetSpacing())
        final_mask_sitk2.SetDirection(sitk_imaget2.GetDirection())
        final_mask_sitk2.SetOrigin(sitk_imaget2.GetOrigin())

        # maskpathname = outmaskpath + "/" + filenames[subsetindex][:-len(imagetype)] + "binarymask.nii.gz"
        # sitk.WriteImage(final_mask_sitk1, maskpathname)
        maskpathname = outmaskpath + "/" + filenames[subsetindex][:-len(imagetype)] + "mutilmask.nii.gz"
        sitk.WriteImage(final_mask_sitk2, maskpathname)


def inferenceMutilVnet3dtestFDI():
    Vnet3d_binary = BinaryVNet3dModel(image_depth=256, image_height=256, image_width=256, image_channel=1, numclass=1,
                                      batch_size=1, loss_name='BinaryCrossEntropyDiceLoss', accum_gradient_iter=4,
                                      use_cuda=use_cuda, inference=True, model_path=r'log/dicebce3d\BinaryVNet3d.pth')

    Vnet3d_mutil = MutilVNet3dModel(image_depth=160, image_height=256, image_width=256, image_channel=1, numclass=35,
                                    batch_size=1, loss_name='MutilCrossEntropyDiceLoss', accum_gradient_iter=4,
                                    use_cuda=use_cuda, inference=True, model_path=r'log/dicece3d/MutilVNet3dModel.pth')
    datapath = r"F:\MedicalData\2024Semi-TeethSeg\3D CBCT\Validation-Public"
    outmaskpath = r'F:\MedicalData\2024Semi-TeethSeg\3D CBCT\Validation-Public-MaskFDI'
    imagetype = ".nii.gz"
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
        # binary segmentation
        sitk_mask_roi = Vnet3d_binary.inference(roi_src_t2, newSize=(256, 256, 256))
        sitk_mask_roi = RemoveSmallConnectedCompont(sitk_mask_roi, 0.1)
        array_mask_roi = sitk.GetArrayFromImage(sitk_mask_roi)
        final_mask_array1 = np.zeros_like(src_array_t2)
        final_mask_array1[z1:z2, y1:y2, x1:x2] = array_mask_roi.copy()
        final_mask_sitk1 = sitk.GetImageFromArray(final_mask_array1.astype('uint8'))
        final_mask_sitk1.SetSpacing(sitk_imaget2.GetSpacing())
        final_mask_sitk1.SetDirection(sitk_imaget2.GetDirection())
        final_mask_sitk1.SetOrigin(sitk_imaget2.GetOrigin())
        # body region segmentation
        organmasksitk = MorphologicalOperation(final_mask_sitk1, 10, "dilate")
        organmasksitk = GetLargestConnectedCompont(organmasksitk)
        organmaskarray = sitk.GetArrayFromImage(organmasksitk)
        z1, z2 = getRangImageRange(organmaskarray, 0)
        y1, y2 = getRangImageRange(organmaskarray, 1)
        x1, x2 = getRangImageRange(organmaskarray, 2)
        src_array_t2 = sitk.GetArrayFromImage(sitk_imaget2)
        roi_src_array_t2 = src_array_t2[z1:z2, y1:y2, x1:x2]
        roi_src_t2 = sitk.GetImageFromArray(roi_src_array_t2)
        roi_src_t2.SetSpacing(sitk_imaget2.GetSpacing())
        roi_src_t2.SetDirection(sitk_imaget2.GetDirection())
        roi_src_t2.SetOrigin(sitk_imaget2.GetOrigin())
        sitk_mask_roi2 = Vnet3d_mutil.inference(roi_src_t2, newSize=(256, 256, 160))
        array_mask_roi2 = sitk.GetArrayFromImage(sitk_mask_roi2)
        final_mask_array2 = np.zeros_like(src_array_t2)
        final_mask_array2[z1:z2, y1:y2, x1:x2] = array_mask_roi2.copy()
        organmasksitk = MorphologicalOperation(final_mask_sitk1, 3, "dilate")
        final_mask_array2[sitk.GetArrayFromImage(organmasksitk) == 0] = 0
        final_mask_array3 = np.zeros_like(final_mask_array2)
        final_mask_array3[final_mask_array2 == 1] = 1
        final_mask_array3[final_mask_array2 == 2] = 11
        final_mask_array3[final_mask_array2 == 3] = 12
        final_mask_array3[final_mask_array2 == 4] = 13
        final_mask_array3[final_mask_array2 == 5] = 14
        final_mask_array3[final_mask_array2 == 6] = 15
        final_mask_array3[final_mask_array2 == 7] = 16
        final_mask_array3[final_mask_array2 == 8] = 17
        final_mask_array3[final_mask_array2 == 9] = 18
        final_mask_array3[final_mask_array2 == 10] = 21
        final_mask_array3[final_mask_array2 == 11] = 22
        final_mask_array3[final_mask_array2 == 12] = 23
        final_mask_array3[final_mask_array2 == 13] = 24
        final_mask_array3[final_mask_array2 == 14] = 25
        final_mask_array3[final_mask_array2 == 15] = 26
        final_mask_array3[final_mask_array2 == 16] = 27
        final_mask_array3[final_mask_array2 == 17] = 28
        final_mask_array3[final_mask_array2 == 18] = 31
        final_mask_array3[final_mask_array2 == 19] = 32
        final_mask_array3[final_mask_array2 == 20] = 33
        final_mask_array3[final_mask_array2 == 21] = 34
        final_mask_array3[final_mask_array2 == 22] = 35
        final_mask_array3[final_mask_array2 == 23] = 36
        final_mask_array3[final_mask_array2 == 24] = 37
        final_mask_array3[final_mask_array2 == 25] = 38
        final_mask_array3[final_mask_array2 == 26] = 41
        final_mask_array3[final_mask_array2 == 27] = 42
        final_mask_array3[final_mask_array2 == 28] = 43
        final_mask_array3[final_mask_array2 == 29] = 44
        final_mask_array3[final_mask_array2 == 30] = 45
        final_mask_array3[final_mask_array2 == 31] = 46
        final_mask_array3[final_mask_array2 == 32] = 47
        final_mask_array3[final_mask_array2 == 33] = 48
        final_mask_array3[final_mask_array2 == 34] = 71

        final_mask_sitk2 = sitk.GetImageFromArray(final_mask_array3.astype('uint8'))
        final_mask_sitk2.SetSpacing(sitk_imaget2.GetSpacing())
        final_mask_sitk2.SetDirection(sitk_imaget2.GetDirection())
        final_mask_sitk2.SetOrigin(sitk_imaget2.GetOrigin())

        maskpathname = outmaskpath + "/" + filenames[subsetindex][6:-len(imagetype)] + "_Mask.nii.gz"
        sitk.WriteImage(final_mask_sitk2, maskpathname)


if __name__ == '__main__':
    # inferenceMutilVnet3dvalid()
    # inferenceMutilVnet3dtest()
    inferenceMutilVnet3dtestFDI()
