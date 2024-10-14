from __future__ import print_function, division
import SimpleITK as sitk
import numpy as np
from dataprocess.utils import file_name_path

imagetype = "label.nii.gz"


def getImageSizeandSpacing(aorticvalve_path):
    """
    get image and spacing
    :return:
    """
    file_path_list = file_name_path(aorticvalve_path, False, True)
    size = []
    spacing = []
    label_list = []
    for subsetindex in range(len(file_path_list)):
        mask_name = file_path_list[subsetindex]
        if imagetype in mask_name:
            mask_gt_file = aorticvalve_path + "/" + mask_name
            src = sitk.ReadImage(mask_gt_file, sitk.sitkUInt8)
            mask = sitk.GetArrayFromImage(src)
            imageSize = src.GetSize()
            imageSpacing = src.GetSpacing()
            size.append(np.array(imageSize))
            spacing.append(np.array(imageSpacing))
            print(np.unique(mask))
            label_list.extend(list(np.unique(mask)))
            print("image name,image size,image spacing:", (mask_name, imageSize, imageSpacing))
    print(np.unique(np.array(label_list)))
    print("median size,median spacing:", (np.median(np.array(size), axis=0), np.median(np.array(spacing), axis=0)))
    print("min size,min spacing:", (np.min(np.array(size), axis=0), np.min(np.array(spacing), axis=0)))
    print("max size,max spacing:", (np.max(np.array(size), axis=0), np.max(np.array(spacing), axis=0)))


if __name__ == "__main__":
    aorticvalve_path = r"F:\MedicalData\2024Semi-TeethSeg\roiprocessCBCT"
    getImageSizeandSpacing(aorticvalve_path)
