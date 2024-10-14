from __future__ import print_function, division
import cv2
from dataprocess.utils import file_name_path
import numpy as np
import json

label_list = []


def extractcontours(mask, label_number):
    binarymask = np.zeros_like(mask)
    binarymask[mask == label_number] = 255
    contours, _ = cv2.findContours(binarymask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        max_contour = max(contours, key=cv2.contourArea)
        max_contour = np.squeeze(max_contour).tolist()
        if len(max_contour) < 10:
            max_contour = []
    else:
        max_contour = []
    return max_contour


def convertmasktojson(mask):
    shapes_list = []
    contour_11 = extractcontours(mask, 1)
    if len(contour_11):
        shapes_list.append({"label": "11", "points": contour_11})
    contour_12 = extractcontours(mask, 2)
    if len(contour_12):
        shapes_list.append({"label": "12", "points": contour_12})
    contour_13 = extractcontours(mask, 3)
    if len(contour_13):
        shapes_list.append({"label": "13", "points": contour_13})
    contour_14 = extractcontours(mask, 4)
    if len(contour_14):
        shapes_list.append({"label": "14", "points": contour_14})
    contour_15 = extractcontours(mask, 5)
    if len(contour_15):
        shapes_list.append({"label": "15", "points": contour_15})
    contour_16 = extractcontours(mask, 6)
    if len(contour_16):
        shapes_list.append({"label": "16", "points": contour_16})
    contour_17 = extractcontours(mask, 7)
    if len(contour_17):
        shapes_list.append({"label": "17", "points": contour_17})
    contour_18 = extractcontours(mask, 8)
    if len(contour_18):
        shapes_list.append({"label": "18", "points": contour_18})
    contour_21 = extractcontours(mask, 9)
    if len(contour_21):
        shapes_list.append({"label": "21", "points": contour_21})
    contour_22 = extractcontours(mask, 10)
    if len(contour_22):
        shapes_list.append({"label": "22", "points": contour_22})
    contour_23 = extractcontours(mask, 11)
    if len(contour_23):
        shapes_list.append({"label": "23", "points": contour_23})
    contour_24 = extractcontours(mask, 12)
    if len(contour_24):
        shapes_list.append({"label": "24", "points": contour_24})
    contour_25 = extractcontours(mask, 13)
    if len(contour_25):
        shapes_list.append({"label": "25", "points": contour_25})
    contour_26 = extractcontours(mask, 14)
    if len(contour_26):
        shapes_list.append({"label": "26", "points": contour_26})
    contour_27 = extractcontours(mask, 15)
    if len(contour_27):
        shapes_list.append({"label": "27", "points": contour_27})
    contour_28 = extractcontours(mask, 16)
    if len(contour_28):
        shapes_list.append({"label": "28", "points": contour_28})
    contour_31 = extractcontours(mask, 17)
    if len(contour_31):
        shapes_list.append({"label": "31", "points": contour_31})
    contour_32 = extractcontours(mask, 18)
    if len(contour_32):
        shapes_list.append({"label": "32", "points": contour_32})
    contour_33 = extractcontours(mask, 19)
    if len(contour_33):
        shapes_list.append({"label": "33", "points": contour_33})
    contour_34 = extractcontours(mask, 20)
    if len(contour_34):
        shapes_list.append({"label": "34", "points": contour_34})
    contour_35 = extractcontours(mask, 21)
    if len(contour_35):
        shapes_list.append({"label": "35", "points": contour_35})
    contour_36 = extractcontours(mask, 22)
    if len(contour_36):
        shapes_list.append({"label": "36", "points": contour_36})
    contour_37 = extractcontours(mask, 23)
    if len(contour_37):
        shapes_list.append({"label": "37", "points": contour_37})
    contour_38 = extractcontours(mask, 24)
    if len(contour_38):
        shapes_list.append({"label": "38", "points": contour_38})
    contour_41 = extractcontours(mask, 25)
    if len(contour_41):
        shapes_list.append({"label": "41", "points": contour_41})
    contour_42 = extractcontours(mask, 26)
    if len(contour_42):
        shapes_list.append({"label": "42", "points": contour_42})
    contour_43 = extractcontours(mask, 27)
    if len(contour_43):
        shapes_list.append({"label": "43", "points": contour_43})
    contour_44 = extractcontours(mask, 28)
    if len(contour_44):
        shapes_list.append({"label": "44", "points": contour_44})
    contour_45 = extractcontours(mask, 29)
    if len(contour_45):
        shapes_list.append({"label": "45", "points": contour_45})
    contour_46 = extractcontours(mask, 30)
    if len(contour_46):
        shapes_list.append({"label": "46", "points": contour_46})
    contour_47 = extractcontours(mask, 31)
    if len(contour_47):
        shapes_list.append({"label": "47", "points": contour_47})
    contour_48 = extractcontours(mask, 32)
    if len(contour_48):
        shapes_list.append({"label": "48", "points": contour_48})
    contour_51 = extractcontours(mask, 33)
    if len(contour_51):
        shapes_list.append({"label": "51", "points": contour_51})
    contour_52 = extractcontours(mask, 34)
    if len(contour_52):
        shapes_list.append({"label": "52", "points": contour_52})
    contour_53 = extractcontours(mask, 35)
    if len(contour_53):
        shapes_list.append({"label": "53", "points": contour_53})
    contour_54 = extractcontours(mask, 36)
    if len(contour_54):
        shapes_list.append({"label": "54", "points": contour_54})
    contour_55 = extractcontours(mask, 37)
    if len(contour_55):
        shapes_list.append({"label": "55", "points": contour_55})
    contour_61 = extractcontours(mask, 38)
    if len(contour_61):
        shapes_list.append({"label": "61", "points": contour_61})
    contour_62 = extractcontours(mask, 39)
    if len(contour_62):
        shapes_list.append({"label": "62", "points": contour_62})
    contour_63 = extractcontours(mask, 40)
    if len(contour_63):
        shapes_list.append({"label": "63", "points": contour_63})
    contour_64 = extractcontours(mask, 41)
    if len(contour_64):
        shapes_list.append({"label": "64", "points": contour_64})
    contour_65 = extractcontours(mask, 42)
    if len(contour_65):
        shapes_list.append({"label": "65", "points": contour_65})
    contour_71 = extractcontours(mask, 43)
    if len(contour_71):
        shapes_list.append({"label": "71", "points": contour_71})
    contour_72 = extractcontours(mask, 44)
    if len(contour_72):
        shapes_list.append({"label": "72", "points": contour_72})
    contour_73 = extractcontours(mask, 45)
    if len(contour_73):
        shapes_list.append({"label": "73", "points": contour_73})
    contour_74 = extractcontours(mask, 46)
    if len(contour_74):
        shapes_list.append({"label": "74", "points": contour_74})
    contour_75 = extractcontours(mask, 47)
    if len(contour_75):
        shapes_list.append({"label": "75", "points": contour_75})
    contour_81 = extractcontours(mask, 48)
    if len(contour_81):
        shapes_list.append({"label": "81", "points": contour_81})
    contour_82 = extractcontours(mask, 49)
    if len(contour_82):
        shapes_list.append({"label": "82", "points": contour_82})
    contour_83 = extractcontours(mask, 50)
    if len(contour_83):
        shapes_list.append({"label": "83", "points": contour_83})
    contour_84 = extractcontours(mask, 51)
    if len(contour_84):
        shapes_list.append({"label": "84", "points": contour_84})
    contour_85 = extractcontours(mask, 52)
    if len(contour_85):
        shapes_list.append({"label": "85", "points": contour_85})

    json_data = {"shapes": shapes_list, "imageHeight": mask.shape[0], "imageWidth": mask.shape[1]}
    return json_data


def convertjsontomask(json_data, image):
    json_shape = json_data['shapes']
    mask = np.zeros_like(image)
    for i in range(len(json_shape)):
        label = json_shape[i]['label']
        label_list.append(int(label))
        contours = json_shape[i]['points']
        opencvcontour = np.array(contours).reshape((len(contours), 1, 2)).astype(np.int)
        if int(label) == 11:
            colorvalue = 1
        if int(label) == 12:
            colorvalue = 2
        if int(label) == 13:
            colorvalue = 3
        if int(label) == 14:
            colorvalue = 4
        if int(label) == 15:
            colorvalue = 5
        if int(label) == 16:
            colorvalue = 6
        if int(label) == 17:
            colorvalue = 7
        if int(label) == 18:
            colorvalue = 8
        if int(label) == 21:
            colorvalue = 9
        if int(label) == 22:
            colorvalue = 10
        if int(label) == 23:
            colorvalue = 11
        if int(label) == 24:
            colorvalue = 12
        if int(label) == 25:
            colorvalue = 13
        if int(label) == 26:
            colorvalue = 14
        if int(label) == 27:
            colorvalue = 15
        if int(label) == 28:
            colorvalue = 16
        if int(label) == 31:
            colorvalue = 17
        if int(label) == 32:
            colorvalue = 18
        if int(label) == 33:
            colorvalue = 19
        if int(label) == 34:
            colorvalue = 20
        if int(label) == 35:
            colorvalue = 21
        if int(label) == 36:
            colorvalue = 22
        if int(label) == 37:
            colorvalue = 23
        if int(label) == 38:
            colorvalue = 24
        if int(label) == 41:
            colorvalue = 25
        if int(label) == 42:
            colorvalue = 26
        if int(label) == 43:
            colorvalue = 27
        if int(label) == 44:
            colorvalue = 28
        if int(label) == 45:
            colorvalue = 29
        if int(label) == 46:
            colorvalue = 30
        if int(label) == 47:
            colorvalue = 31
        if int(label) == 48:
            colorvalue = 32
        if int(label) == 51:
            colorvalue = 33
        if int(label) == 52:
            colorvalue = 34
        if int(label) == 53:
            colorvalue = 35
        if int(label) == 54:
            colorvalue = 36
        if int(label) == 55:
            colorvalue = 37
        if int(label) == 61:
            colorvalue = 38
        if int(label) == 62:
            colorvalue = 39
        if int(label) == 63:
            colorvalue = 40
        if int(label) == 64:
            colorvalue = 41
        if int(label) == 65:
            colorvalue = 42
        if int(label) == 71:
            colorvalue = 43
        if int(label) == 72:
            colorvalue = 44
        if int(label) == 73:
            colorvalue = 45
        if int(label) == 74:
            colorvalue = 46
        if int(label) == 75:
            colorvalue = 47
        if int(label) == 81:
            colorvalue = 48
        if int(label) == 82:
            colorvalue = 49
        if int(label) == 83:
            colorvalue = 50
        if int(label) == 84:
            colorvalue = 51
        if int(label) == 85:
            colorvalue = 52
        cv2.drawContours(mask, [opencvcontour], -1, colorvalue, -1)
    return mask


def x_ray_segmentation(input_path_image, output_path):
    input_path_image_dir = input_path_image + '/Images'
    input_path_mask_dir = input_path_image + '/Masks'
    all_imagefiles_list = file_name_path(input_path_image_dir, False, True)
    all_jsonfiles_list = file_name_path(input_path_mask_dir, False, True)
    for i in range(len(all_jsonfiles_list)):
        jsonfile = input_path_mask_dir + '/' + all_jsonfiles_list[i]
        imagefile = input_path_image_dir + '/' + all_imagefiles_list[i]
        src_image = cv2.imread(imagefile, 0)
        # 读取 JSON 文件
        with open(jsonfile, 'r') as file:
            json_data = json.load(file)
        mask = convertjsontomask(json_data, src_image)
        cv2.imwrite(output_path + '/Image/' + all_imagefiles_list[i][:-3] + 'bmp', src_image)
        cv2.imwrite(output_path + '/Mask/' + all_imagefiles_list[i][:-3] + 'bmp', mask)

    print(np.unique(np.array(label_list)))


def x_ray_tran_unlabel_segmentation(input_path_image, output_path):
    input_path_image_dir = input_path_image + '/Train-Unlabeled'
    input_path_mask_dir = input_path_image + '/Train-Unlabeled-Mask'
    all_imagefiles_list = file_name_path(input_path_image_dir, False, True)
    all_jsonfiles_list = file_name_path(input_path_mask_dir, False, True)
    for i in range(len(all_jsonfiles_list)):
        jsonfile = input_path_mask_dir + '/' + all_jsonfiles_list[i]
        imagefile = input_path_image_dir + '/' + all_imagefiles_list[i]
        src_image = cv2.imread(imagefile, 0)
        src_mask = cv2.imread(jsonfile, 0)
        cv2.imwrite(output_path + '/Image/' + all_imagefiles_list[i][:-3] + 'bmp', src_image)
        cv2.imwrite(output_path + '/Mask/' + all_imagefiles_list[i][:-3] + 'bmp', src_mask)


if __name__ == "__main__":
    # x_ray_segmentation(r"F:\MedicalData\2024Semi-TeethSeg\2D XRAY\Train-Labeled",
    #                    r"F:\MedicalData\2024Semi-TeethSeg\processXRAY")

    x_ray_tran_unlabel_segmentation(r"F:\MedicalData\2024Semi-TeethSeg\2D XRAY",
                                    r"E:\MedicalData\2024Semi-TeethSeg\2D\unlabel_train")
