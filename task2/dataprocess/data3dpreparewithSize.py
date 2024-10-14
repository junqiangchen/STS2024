from __future__ import print_function, division
import os
import SimpleITK as sitk
import numpy as np
from dataprocess.utils import file_name_path, resize_image_itkwithsize, ConvertitkTrunctedValue

image_dir = "Image"
mask_dir = "Mask"
imaget2_pre = '_image.nii.gz'
mask_pre = '_label.nii.gz'


def preparesampling3dtraindata(datapath, trainImage, trainMask, shape=(96, 96, 96)):
    newSize = shape
    all_files = file_name_path(datapath, False, True)

    mask_files_list = []
    for index in range(len(all_files)):
        if mask_pre in all_files[index]:
            mask_files_list.append(all_files[index])

    for subsetindex in range(len(mask_files_list)):
        mask_name = mask_files_list[subsetindex]
        image_name = mask_name[:-len(mask_pre)] + imaget2_pre
        mask_gt_file = datapath + "/" + mask_name
        masksegsitk = sitk.ReadImage(mask_gt_file)
        imaget2_file = datapath + "/" + image_name
        imaget2sitk = sitk.ReadImage(imaget2_file)

        _, resizeimaget2 = resize_image_itkwithsize(imaget2sitk, newSize, imaget2sitk.GetSize(),
                                                    sitk.sitkLinear)
        _, resizemask = resize_image_itkwithsize(masksegsitk, newSize, masksegsitk.GetSize(),
                                                 sitk.sitkNearestNeighbor)
        resizeimaget2 = ConvertitkTrunctedValue(resizeimaget2, 2000, 0, 'meanstd')
        resizeimagearrayt2 = sitk.GetArrayFromImage(resizeimaget2)
        resizemaskarray = sitk.GetArrayFromImage(resizemask)
        # sitk.WriteImage(sitk.GetImageFromArray(resizeimagearrayt2), 'resizeimagearrayt2.nii.gz')
        # sitk.WriteImage(sitk.GetImageFromArray(resizemaskarray), 'resizemask.nii.gz')
        # step 3 get subimages and submasks
        if not os.path.exists(trainImage):
            os.makedirs(trainImage)
        if not os.path.exists(trainMask):
            os.makedirs(trainMask)
        filepath1 = trainImage + "\\" + str(subsetindex) + ".npy"
        np.save(filepath1, resizeimagearrayt2)
        filepath = trainMask + "\\" + str(subsetindex) + ".npy"
        print(mask_name)
        print(np.unique(resizemaskarray))
        np.save(filepath, resizemaskarray.astype('uint8'))


def preparetraindata(src_train_path, source_process_path, newSize=(160, 160, 160)):
    """
    :return:
    """
    outputimagepath = source_process_path + "/" + image_dir
    outputlabelpath = source_process_path + "/" + mask_dir
    if not os.path.exists(outputimagepath):
        os.makedirs(outputimagepath)
    if not os.path.exists(outputlabelpath):
        os.makedirs(outputlabelpath)
    preparesampling3dtraindata(src_train_path, outputimagepath, outputlabelpath, newSize)


def preparevalidationdata(src_train_path, source_process_path, newSize=(160, 160, 160)):
    """
    :return:
    """
    outputimagepath = source_process_path + "/" + image_dir
    outputlabelpath = source_process_path + "/" + mask_dir
    if not os.path.exists(outputimagepath):
        os.makedirs(outputimagepath)
    if not os.path.exists(outputlabelpath):
        os.makedirs(outputlabelpath)
    preparesampling3dtraindata(src_train_path, outputimagepath, outputlabelpath, newSize)


if __name__ == "__main__":
    # preparetraindata(src_train_path=r"F:\MedicalData\2024Semi-TeethSeg\roiprocessCBCT\train",
    #                  source_process_path=r"E:\MedicalData\2024Semi-TeethSeg\3D\train",
    #                  newSize=(256, 256, 160))
    # preparevalidationdata(src_train_path=r"F:\MedicalData\2024Semi-TeethSeg\roiprocessCBCT\validation",
    #                       source_process_path=r"E:\MedicalData\2024Semi-TeethSeg\3D\validation",
    #                       newSize=(256, 256, 160))
    preparetraindata(src_train_path=r"F:\MedicalData\2024Semi-TeethSeg\roiprocessCBCT\unlabeltrain",
                     source_process_path=r"E:\MedicalData\2024Semi-TeethSeg\3D\unlabel_train",
                     newSize=(256, 256, 160))
