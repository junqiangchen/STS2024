from __future__ import print_function, division
import os
import numpy as np
import pandas as pd
import SimpleITK as sitk
import cv2


def RemoveSmallConnectedCompont2d(binaryimage, ratio=0.01):
    image = binaryimage.astype('uint8')
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(image, connectivity=4)
    sizes = stats[:, -1]

    max_size = -100
    for i in range(1, nb_components):
        if sizes[i] > max_size:
            max_size = sizes[i]
    max_label_list = []
    for i in range(1, nb_components):
        if sizes[i] > max_size * ratio:
            max_label_list.append(i)

    img2 = np.zeros(output.shape, dtype='uint8')
    for i in range(len(max_label_list)):
        img2[output == max_label_list[i]] = 1
    return img2


def RemoveSmallConnectedCompont(binarysitk_image, ratio=0.01):
    """
    get 3dlargest region
    :param sitk_maskimg:binary itk image
    :return: largest region binary image
    """
    cc = sitk.ConnectedComponent(binarysitk_image)
    stats = sitk.LabelIntensityStatisticsImageFilter()
    stats.SetGlobalDefaultNumberOfThreads(8)
    stats.Execute(cc, binarysitk_image)
    labellist = []
    maxvolume = -1
    for l in stats.GetLabels():
        size = stats.GetPhysicalSize(l)
        if maxvolume < size:
            maxvolume = size
    for l in stats.GetLabels():
        size = stats.GetPhysicalSize(l)
        if size > maxvolume * ratio:
            labellist.append(l)
    labelmaskimage = sitk.GetArrayFromImage(cc)
    outmask = np.zeros_like(labelmaskimage)
    for index in range(len(labellist)):
        outmask[labelmaskimage == labellist[index]] = 1
    outmasksitk = sitk.GetImageFromArray(outmask)
    outmasksitk.SetSpacing(binarysitk_image.GetSpacing())
    outmasksitk.SetDirection(binarysitk_image.GetDirection())
    outmasksitk.SetOrigin(binarysitk_image.GetOrigin())
    return outmasksitk


def GetConnectedCompontBoundingbox(binarysitk_image):
    cc = sitk.ConnectedComponent(binarysitk_image)
    stats = sitk.LabelIntensityStatisticsImageFilter()
    stats.SetGlobalDefaultNumberOfThreads(8)
    stats.Execute(cc, binarysitk_image)
    lsif = sitk.LabelShapeStatisticsImageFilter()
    lsif.Execute(cc)
    boundingBoxs = []
    for l in stats.GetLabels():
        boundingBoxs.append(np.array(lsif.GetBoundingBox(l)))
    return boundingBoxs


def GetConnectedCompontStatistics(binarysitk_image):
    cc = sitk.ConnectedComponent(binarysitk_image)
    stats = sitk.LabelIntensityStatisticsImageFilter()
    stats.SetGlobalDefaultNumberOfThreads(8)
    stats.Execute(cc, binarysitk_image)
    lsif = sitk.LabelShapeStatisticsImageFilter()
    lsif.Execute(cc)
    labels = []
    boundingBoxs = []
    volumes = []
    centers = []
    for l in stats.GetLabels():
        labels.append(l)
        boundingBoxs.append(np.array(lsif.GetBoundingBox(l)))
        volumes.append(lsif.GetPhysicalSize(l))
        centers.append(np.array(lsif.GetCentroid(l)))
    return labels, boundingBoxs, volumes, centers


def GetLargestConnectedCompontBoundingbox(binarysitk_image):
    lsif = sitk.LabelShapeStatisticsImageFilter()
    lsif.Execute(binarysitk_image)
    boundingBox = np.array(lsif.GetBoundingBox(1))  # [xstart, ystart, zstart, xsize, ysize, zsize]
    return boundingBox


def GetLargestConnectedCompont(binarysitk_image):
    """
    get 3dlargest region
    :param sitk_maskimg:binary itk image
    :return: largest region binary image
    """
    cc = sitk.ConnectedComponent(binarysitk_image != 0)
    stats = sitk.LabelIntensityStatisticsImageFilter()
    stats.SetGlobalDefaultNumberOfThreads(8)
    stats.Execute(cc, binarysitk_image)
    maxlabel = 0
    maxsize = 0
    for l in stats.GetLabels():
        size = stats.GetPhysicalSize(l)
        if maxsize < size:
            maxlabel = l
            maxsize = size
    labelmaskimage = sitk.GetArrayFromImage(cc)
    outmask = labelmaskimage.copy()
    outmask[labelmaskimage == maxlabel] = 1
    outmask[labelmaskimage != maxlabel] = 0
    outmasksitk = sitk.GetImageFromArray(outmask)
    outmasksitk.SetSpacing(binarysitk_image.GetSpacing())
    outmasksitk.SetDirection(binarysitk_image.GetDirection())
    outmasksitk.SetOrigin(binarysitk_image.GetOrigin())
    return outmasksitk


def MorphologicalOperation(sitk_maskimg, kernelsize, name='open'):
    """
    morphological operation
    :param sitk_maskimg:
    :param kernelsize:
    :param name:operation name
    :return:binary image
    """
    if name == 'open':
        morphoimage = sitk.BinaryMorphologicalOpening(sitk_maskimg != 0, [kernelsize] * sitk_maskimg.GetDimension())
        return morphoimage
    if name == 'close':
        morphoimage = sitk.BinaryMorphologicalClosing(sitk_maskimg != 0, [kernelsize] * sitk_maskimg.GetDimension())
        return morphoimage
    if name == 'dilate':
        morphoimage = sitk.BinaryDilate(sitk_maskimg != 0, [kernelsize] * sitk_maskimg.GetDimension())
        return morphoimage
    if name == 'erode':
        morphoimage = sitk.BinaryErode(sitk_maskimg != 0, [kernelsize] * sitk_maskimg.GetDimension())
        return morphoimage


def getRangImageRange(image, index=0):
    """
    :param image:
    :return:rang of image depth
    """
    startposition = 0
    endposition = 0
    for z in range(0, image.shape[index], 1):
        if index == 0:
            notzeroflag = np.max(image[z, :, :])
        elif index == 1:
            notzeroflag = np.max(image[:, z, :])
        elif index == 2:
            notzeroflag = np.max(image[:, :, z])
        if notzeroflag:
            startposition = z
            break
    for z in range(image.shape[index] - 1, -1, -1):
        if index == 0:
            notzeroflag = np.max(image[z, :, :])
        elif index == 1:
            notzeroflag = np.max(image[:, z, :])
        elif index == 2:
            notzeroflag = np.max(image[:, :, z])
        if notzeroflag:
            endposition = z
            break
    return startposition, endposition


def resize_image_itkwithsize(itkimage, newSize, originSize, resamplemethod=sitk.sitkNearestNeighbor):
    """
    image resize withe sitk resampleImageFilter
    :param itkimage:
    :param newSize:such as [1,1,1]
    :param resamplemethod:
    :return:
    """
    resampler = sitk.ResampleImageFilter()
    originSize = np.array(originSize)
    newSize = np.array(newSize)
    factor = originSize / newSize
    originSpcaing = itkimage.GetSpacing()
    newSpacing = factor * originSpcaing
    resampler.SetReferenceImage(itkimage)
    resampler.SetOutputSpacing(newSpacing.tolist())
    resampler.SetSize(newSize.tolist())
    resampler.SetTransform(sitk.Transform(3, sitk.sitkIdentity))
    resampler.SetInterpolator(resamplemethod)
    itkimgResampled = resampler.Execute(itkimage)
    imgResampled = sitk.GetArrayFromImage(itkimgResampled)
    return imgResampled, itkimgResampled


def resize_image_itk(itkimage, newSpacing, originSpcaing, resamplemethod=sitk.sitkNearestNeighbor):
    """
    image resize withe sitk resampleImageFilter
    :param itkimage:
    :param newSpacing:such as [1,1,1]
    :param resamplemethod:
    :return:
    """
    newSpacing = np.array(newSpacing, float)
    # originSpcaing = itkimage.GetSpacing()
    resampler = sitk.ResampleImageFilter()
    originSize = itkimage.GetSize()
    factor = newSpacing / originSpcaing
    newSize = originSize / factor
    newSize = newSize.astype(np.int)
    resampler.SetReferenceImage(itkimage)
    resampler.SetOutputSpacing(newSpacing.tolist())
    resampler.SetSize(newSize.tolist())
    resampler.SetTransform(sitk.Transform(3, sitk.sitkIdentity))
    resampler.SetInterpolator(resamplemethod)
    itkimgResampled = resampler.Execute(itkimage)
    imgResampled = sitk.GetArrayFromImage(itkimgResampled)
    return imgResampled, itkimgResampled


def ConvertitkTrunctedValue(image, upper=200, lower=-200, normalize='None'):
    """
    load files,set truncted value range and normalization 0-255
    :param filename:
    :param upper:
    :param lower:
    :return:
    """
    # 1,tructed outside of liver value
    srcitkimage = sitk.Cast(image, sitk.sitkFloat32)
    srcitkimagearray = sitk.GetArrayFromImage(srcitkimage)
    srcitkimagearray[srcitkimagearray > upper] = upper
    srcitkimagearray[srcitkimagearray < lower] = lower
    # 2,get tructed outside of liver value image
    sitktructedimage = sitk.GetImageFromArray(srcitkimagearray)
    origin = np.array(srcitkimage.GetOrigin())
    spacing = np.array(srcitkimage.GetSpacing())
    sitktructedimage.SetSpacing(spacing)
    sitktructedimage.SetOrigin(origin)
    itkimage = sitk.Cast(sitktructedimage, sitk.sitkFloat32)
    # 3 normalization value to 0-255
    if normalize == 'maxmin':
        rescalFilt = sitk.RescaleIntensityImageFilter()
        rescalFilt.SetOutputMaximum(1)
        rescalFilt.SetOutputMinimum(0)
        rescalFilt.SetGlobalDefaultNumberOfThreads(8)
        itkimage = rescalFilt.Execute(itkimage)
    if normalize == "meanstd":
        normalizeFilt = sitk.NormalizeImageFilter()
        normalizeFilt.SetNumberOfThreads(8)
        itkimage = normalizeFilt.Execute(itkimage)
    return itkimage


def normalize(slice, bottom=95, down=5):
    """
    normalize image with mean and std for regionnonzero,and clip the value into range
    :param slice:
    :param bottom:
    :param down:
    :return:
    """
    b = np.percentile(slice, bottom)
    t = np.percentile(slice, down)
    slice = np.clip(slice, t, b)

    image_nonzero = slice[np.nonzero(slice)]
    if np.std(slice) == 0 or np.std(image_nonzero) == 0:
        return slice
    else:
        tmp = (slice - np.mean(image_nonzero)) / np.std(image_nonzero)
        # since the range of intensities is between 0 and 5000 ,
        # the min in the normalized slice corresponds to 0 intensity in unnormalized slice
        # the min is replaced with -9 just to keep track of 0 intensities
        # so that we can discard those intensities afterwards when sampling random patches
        # tmp[tmp == tmp.min()] = -9
        return tmp


def calcu_dice(Y_pred, Y_gt, K=255):
    """
    calculate two input dice value
    :param Y_pred:
    :param Y_gt:
    :param K:
    :return:
    """
    intersection = 2 * np.sum(Y_pred[Y_gt == K])
    denominator = np.sum(Y_pred) + np.sum(Y_gt) + 1e-5
    loss = (intersection / denominator)
    return loss


def file_name_path(file_dir, dir=True, file=False):
    """
    get root path,sub_dirs,all_sub_files
    :param file_dir:
    :return: dir or file
    """
    for root, dirs, files in os.walk(file_dir):
        if len(dirs) and dir:
            print("sub_dirs:", dirs)
            return dirs
        if len(files) and file:
            print("files:", files)
            return files


def check_trained_data(csv_path, ratio=0.001):
    csvdata = pd.read_csv(csv_path)
    maskdata = csvdata.iloc[:, 1].values
    count = 0
    for num in range(len(maskdata)):
        mask = np.load(maskdata[num])
        labelpixel_count = (mask != 0).sum()
        backgroundpixel_count = (mask == 0).sum()
        pixel_ratio = labelpixel_count / backgroundpixel_count
        if backgroundpixel_count != 0 and pixel_ratio > ratio:
            count = count + 1
    print(count / len(maskdata))


def save_file2csvregression(file_dir, file_name, csv_gt):
    """
    save file path to csv
    :param file_dir:preprocess data path
    :param file_name:output csv name
    :return:
    """
    csv_gt_data = pd.read_csv(csv_gt)
    PatientIDs = csv_gt_data.iloc[:, 0].values
    RFSvalues = csv_gt_data.iloc[:, 2].values
    labels = file_name_path(file_dir, True, False)

    RFSlist = []
    for i in range(len(labels)):
        for j in range(len(PatientIDs)):
            if labels[i] == PatientIDs[j]:
                RFSlist.append(RFSvalues[j])

    out = open(file_name, 'w')
    for index in range(len(labels)):
        out_file_image_paths = file_dir + "/" + labels[index]
        out.writelines(str(RFSlist[index]) + "," + out_file_image_paths + "\n")


def save_file2csv(file_dir, file_name):
    """
    save file path to csv,this is for segmentation
    :param file_dir:preprocess data path
    :param file_name:output csv name
    :return:
    """
    out = open(file_name, 'w')
    image = "Image"
    mask = "Mask"
    file_image_dir = file_dir + "/" + image
    file_mask_dir = file_dir + "/" + mask
    file_paths = file_name_path(file_image_dir, dir=False, file=True)
    for index in range(len(file_paths)):
        out_file_image_path = file_image_dir + "/" + file_paths[index]
        out_file_mask_path = file_mask_dir + "/" + file_paths[index]
        out.writelines(out_file_image_path + "," + out_file_mask_path + "\n")


def save_file2csvv1(file_dir, file_name):
    """
    save file path to csv,this is for segmentation
    :param file_dir:preprocess data path
    :param file_name:output csv name
    :return:
    """
    out = open(file_name, 'w')
    image = "image"
    mask = "label"
    file_image_dir = file_dir + "/" + image
    file_mask_dir = file_dir + "/" + mask
    file_paths = file_name_path(file_image_dir, dir=False, file=True)
    for index in range(len(file_paths)):
        out_file_image_path = file_image_dir + "/" + file_paths[index]
        out_file_mask_path = file_mask_dir + "/" + file_paths[index]
        out.writelines(out_file_image_path + "," + out_file_mask_path + "\n")


if __name__ == "__main__":
    # save_file2csv(r"E:\MedicalData\2024Semi-TeethSeg\3D\train", "data/train3d.csv")
    # save_file2csv(r"E:\MedicalData\2024Semi-TeethSeg\3D\validation", "data/valid3d.csv")
    # save_file2csv(r"E:\MedicalData\2024Semi-TeethSeg\3D\augtrain", "data/augtrain3d.csv")
    # save_file2csv(r"F:\MedicalData\2024Semi-TeethSeg\processXRAY", "data/alldata2d.csv")
    # save_file2csv(r"E:\MedicalData\2024Semi-TeethSeg\2D\augtrain", "data/augtrain2d.csv")
    # save_file2csv(r"E:\MedicalData\2024Semi-TeethSeg\2D\unlabel_train", "data/unlabel_train2d.csv")
    save_file2csv(r"E:\MedicalData\2024Semi-TeethSeg\3D\unlabel_train", "data/unlabel_train3d.csv")
