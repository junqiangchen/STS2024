import os
import numpy as np
import SimpleITK as sitk

# TODO: import your model
from model import *
from dataprocess.utils import GetLargestConnectedCompontBoundingbox, RemoveSmallConnectedCompont, \
    GetLargestConnectedCompont, MorphologicalOperation, getRangImageRange
from dataprocess.STSdataprocess import getbrain

INPUT_DIR = '/inputs'
OUTPUT_DIR = '/outputs'

# INPUT_DIR = r'F:\MedicalData\(ok)2024Semi-TeethSeg\docker_test\task2\inputs'
# OUTPUT_DIR = r'F:\MedicalData\(ok)2024Semi-TeethSeg\docker_test\task2\outputs'


def inferenceMutilVnet3dtestFDI(sitk_imaget2):
    Vnet3d_binary = BinaryVNet3dModel(image_depth=256, image_height=256, image_width=256, image_channel=1, numclass=1,
                                      batch_size=1, loss_name='BinaryCrossEntropyDiceLoss', accum_gradient_iter=4,
                                      use_cuda=True, inference=True, model_path=r'BinaryVNet3d.pth')

    Vnet3d_mutil = MutilVNet3dModel(image_depth=160, image_height=256, image_width=256, image_channel=1, numclass=35,
                                    batch_size=1, loss_name='MutilCrossEntropyDiceLoss', accum_gradient_iter=4,
                                    use_cuda=True, inference=True, model_path=r'MutilVNet3dModel.pth')
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
    return final_mask_sitk2


def main():
    case_name = os.listdir(INPUT_DIR)[0]
    print(case_name)
    # load image as numpy array
    case_image = sitk.ReadImage(os.path.join(INPUT_DIR, case_name))
    sitk_prediction = inferenceMutilVnet3dtestFDI(case_image)
    case_tag = case_name.split('.')[0]
    sitk.WriteImage(sitk_prediction, os.path.join(OUTPUT_DIR, '%s_Mask.nii.gz' % case_tag))


if __name__ == "__main__":
    main()
