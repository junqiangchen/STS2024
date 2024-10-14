from __future__ import print_function, division
import SimpleITK as sitk
from dataprocess.utils import file_name_path, MorphologicalOperation, GetLargestConnectedCompont, \
    GetLargestConnectedCompontBoundingbox, getRangImageRange
import numpy as np

imaget2_pre = '_image.nii.gz'
mask_pre = '_label.nii.gz'
imagetype = ".nii.gz"


def getbrain(imagesitk):
    minmaxenhancevalue = sitk.MinimumMaximumImageFilter()
    minmaxenhancevalue.Execute(imagesitk)
    max_value = minmaxenhancevalue.GetMaximum()
    bodymasksitk = sitk.BinaryThreshold(imagesitk, -850, max_value)
    bodymasksitk = MorphologicalOperation(bodymasksitk, 3)
    bodymasksitk = GetLargestConnectedCompont(bodymasksitk)
    return bodymasksitk


def roiprocess(input_image_dir, output_path):
    input_image_path = input_image_dir + '/' + "image"
    input_mask_path = input_image_dir + '/' + "label"
    filenames = file_name_path(input_image_path, False, True)
    for subsetindex in range(len(filenames)):
        imagepath = input_image_path + '/' + filenames[subsetindex]
        imagesitk = sitk.ReadImage(imagepath)
        maskpath = input_mask_path + '/' + filenames[subsetindex]
        masksitk = sitk.ReadImage(maskpath)
        # brain region segmentation
        brainmasksitk = getbrain(imagesitk)
        bodyboundingbox = GetLargestConnectedCompontBoundingbox(brainmasksitk)
        x1, y1, z1, x2, y2, z2 = bodyboundingbox[0], bodyboundingbox[1], \
                                 bodyboundingbox[2], bodyboundingbox[0] + bodyboundingbox[3], \
                                 bodyboundingbox[1] + bodyboundingbox[4], bodyboundingbox[2] + \
                                 bodyboundingbox[5]
        src_array_t2 = sitk.GetArrayFromImage(imagesitk)
        maskarray = sitk.GetArrayFromImage(masksitk)

        roi_src_array_t2 = src_array_t2[z1:z2, y1:y2, x1:x2]
        roi_mask_array = maskarray[z1:z2, y1:y2, x1:x2]
        print(np.unique(roi_mask_array))

        roi_src_t2w = sitk.GetImageFromArray(roi_src_array_t2)
        roi_src_t2w.SetSpacing(imagesitk.GetSpacing())
        roi_src_t2w.SetDirection(imagesitk.GetDirection())
        roi_src_t2w.SetOrigin(imagesitk.GetOrigin())

        roi_mask = sitk.GetImageFromArray(roi_mask_array.astype('uint8'))
        roi_mask.SetSpacing(masksitk.GetSpacing())
        roi_mask.SetDirection(masksitk.GetDirection())
        roi_mask.SetOrigin(masksitk.GetOrigin())

        imaget2_file = output_path + "/" + filenames[subsetindex][:-len(imagetype)] + imaget2_pre
        mask_file = output_path + "/" + filenames[subsetindex][:-len(imagetype)] + mask_pre
        sitk.WriteImage(roi_src_t2w, imaget2_file)
        sitk.WriteImage(roi_mask, mask_file)


def Processroidatatask2(input_path_image, output_path):
    input_path_image_dir = input_path_image + '/Images'
    input_path_mask_dir = input_path_image + '/Masks'
    all_files_list = file_name_path(input_path_image_dir, False, True)
    all_maskfiles_list = file_name_path(input_path_mask_dir, False, True)
    for subsetindex in range(len(all_files_list)):
        imagename = all_files_list[subsetindex]
        maskname = all_maskfiles_list[subsetindex]
        imagepath = input_path_image_dir + "/" + imagename
        maskpath = input_path_mask_dir + "/" + maskname
        imagesitk = sitk.ReadImage(imagepath)
        masksitk = sitk.ReadImage(maskpath)
        # body region segmentation
        organmasksitk = MorphologicalOperation(masksitk, 10, "dilate")
        organmasksitk = GetLargestConnectedCompont(organmasksitk)
        organmaskarray = sitk.GetArrayFromImage(organmasksitk)
        z1, z2 = getRangImageRange(organmaskarray, 0)
        y1, y2 = getRangImageRange(organmaskarray, 1)
        x1, x2 = getRangImageRange(organmaskarray, 2)
        src_array = sitk.GetArrayFromImage(imagesitk)
        maskarraysrc = sitk.GetArrayFromImage(masksitk)
        maskarray = np.zeros_like(maskarraysrc)
        maskarray[maskarraysrc == 1] = 1
        maskarray[maskarraysrc == 11] = 2
        maskarray[maskarraysrc == 12] = 3
        maskarray[maskarraysrc == 13] = 4
        maskarray[maskarraysrc == 14] = 5
        maskarray[maskarraysrc == 15] = 6
        maskarray[maskarraysrc == 16] = 7
        maskarray[maskarraysrc == 17] = 8
        maskarray[maskarraysrc == 18] = 9
        maskarray[maskarraysrc == 21] = 10
        maskarray[maskarraysrc == 22] = 11
        maskarray[maskarraysrc == 23] = 12
        maskarray[maskarraysrc == 24] = 13
        maskarray[maskarraysrc == 25] = 14
        maskarray[maskarraysrc == 26] = 15
        maskarray[maskarraysrc == 27] = 16
        maskarray[maskarraysrc == 28] = 17
        maskarray[maskarraysrc == 31] = 18
        maskarray[maskarraysrc == 32] = 19
        maskarray[maskarraysrc == 33] = 20
        maskarray[maskarraysrc == 34] = 21
        maskarray[maskarraysrc == 35] = 22
        maskarray[maskarraysrc == 36] = 23
        maskarray[maskarraysrc == 37] = 24
        maskarray[maskarraysrc == 38] = 25
        maskarray[maskarraysrc == 41] = 26
        maskarray[maskarraysrc == 42] = 27
        maskarray[maskarraysrc == 43] = 28
        maskarray[maskarraysrc == 44] = 29
        maskarray[maskarraysrc == 45] = 30
        maskarray[maskarraysrc == 46] = 31
        maskarray[maskarraysrc == 47] = 32
        maskarray[maskarraysrc == 48] = 33
        maskarray[maskarraysrc == 71] = 34
        roi_src_array = src_array[z1:z2, y1:y2, x1:x2]
        roi_mask_array = maskarray[z1:z2, y1:y2, x1:x2]
        print(np.unique(roi_mask_array))
        roi_src = sitk.GetImageFromArray(roi_src_array)
        roi_src.SetSpacing(imagesitk.GetSpacing())
        roi_src.SetDirection(imagesitk.GetDirection())
        roi_src.SetOrigin(imagesitk.GetOrigin())
        roi_mask = sitk.GetImageFromArray(roi_mask_array)
        roi_mask.SetSpacing(masksitk.GetSpacing())
        roi_mask.SetDirection(masksitk.GetDirection())
        roi_mask.SetOrigin(masksitk.GetOrigin())

        image_file = output_path + "/" + imagename[:-len(imagetype)] + imaget2_pre
        mask_file = output_path + "/" + imagename[:-len(imagetype)] + mask_pre
        sitk.WriteImage(roi_src, image_file)
        sitk.WriteImage(roi_mask, mask_file)


def Processroidatatask2unlabel(input_path_image, output_path):
    input_path_image_dir = input_path_image + '/Train-Unlabeled-update'
    input_path_mask_dir = input_path_image + '/Train-Unlabeled-update-Mask'
    all_files_list = file_name_path(input_path_image_dir, False, True)
    all_maskfiles_list = file_name_path(input_path_mask_dir, False, True)
    for subsetindex in range(len(all_files_list)):
        imagename = all_files_list[subsetindex]
        maskname = all_maskfiles_list[subsetindex]
        imagepath = input_path_image_dir + "/" + imagename
        maskpath = input_path_mask_dir + "/" + maskname
        imagesitk = sitk.ReadImage(imagepath)
        masksitk = sitk.ReadImage(maskpath)
        # body region segmentation
        organmasksitk = MorphologicalOperation(masksitk, 10, "dilate")
        organmasksitk = GetLargestConnectedCompont(organmasksitk)
        organmaskarray = sitk.GetArrayFromImage(organmasksitk)
        z1, z2 = getRangImageRange(organmaskarray, 0)
        y1, y2 = getRangImageRange(organmaskarray, 1)
        x1, x2 = getRangImageRange(organmaskarray, 2)
        src_array = sitk.GetArrayFromImage(imagesitk)
        maskarraysrc = sitk.GetArrayFromImage(masksitk)
        maskarray = maskarraysrc.copy()
        roi_src_array = src_array[z1:z2, y1:y2, x1:x2]
        roi_mask_array = maskarray[z1:z2, y1:y2, x1:x2]
        print(np.unique(roi_mask_array))
        roi_src = sitk.GetImageFromArray(roi_src_array)
        roi_src.SetSpacing(imagesitk.GetSpacing())
        roi_src.SetDirection(imagesitk.GetDirection())
        roi_src.SetOrigin(imagesitk.GetOrigin())
        roi_mask = sitk.GetImageFromArray(roi_mask_array)
        roi_mask.SetSpacing(masksitk.GetSpacing())
        roi_mask.SetDirection(masksitk.GetDirection())
        roi_mask.SetOrigin(masksitk.GetOrigin())

        image_file = output_path + "/" + imagename[:-len(imagetype)] + imaget2_pre
        mask_file = output_path + "/" + imagename[:-len(imagetype)] + mask_pre
        sitk.WriteImage(roi_src, image_file)
        sitk.WriteImage(roi_mask, mask_file)


if __name__ == "__main__":
    # roiprocess(r"F:\MedicalData\2023STS3D\download\labelled", r"F:\MedicalData\2023STS3D\roiprocess")
    # roiprocess(r"F:\MedicalData\2023STS3D\download\testpdlabelled", r"F:\MedicalData\2023STS3D\roiprocess")
    # Processroidatatask2(r"F:\MedicalData\2024Semi-TeethSeg\3D CBCT\Train-Labeled-update",
    #                     r"F:\MedicalData\2024Semi-TeethSeg\roiprocessCBCT")
    Processroidatatask2unlabel(r"F:\MedicalData\2024Semi-TeethSeg\3D CBCT",
                               r"F:\MedicalData\2024Semi-TeethSeg\roiprocessCBCT\unlabeltrain")
