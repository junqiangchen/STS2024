import os
from dataprocess.utils import RemoveSmallConnectedCompont2d
from dataprocess.STS2ddataprocess import convertmasktojson
import json
from model import *
import cv2

INPUT_DIR = '/inputs'
OUTPUT_DIR = '/outputs'

# INPUT_DIR = r'F:\MedicalData\(ok)2024Semi-TeethSeg\docker_test\task1\inputs'
# OUTPUT_DIR = r'F:\MedicalData\(ok)2024Semi-TeethSeg\docker_test\task1\outputs'


def inferenceMutilVnet2dtestjsonv2(image):
    """
            直接对牙齿进行细分割，然后根据细分割的结果去除小目标后得到分割结果，
            效果比用STS2023年的数据先分割牙齿区域，然后在牙齿区域中分割牙齿细分割。
            """
    Vnet2d2 = MutilVNet2dModel(image_height=1024, image_width=1024, image_channel=1, numclass=53,
                               batch_size=4, loss_name='MutilCrossEntropyDiceLoss', accum_gradient_iter=1,
                               use_cuda=True, inference=True, model_path=r'MutilVNet2dModel.pth')
    mask_pd = Vnet2d2.inference(image)
    mask_pd_binary = mask_pd.copy()
    mask_pd_binary[mask_pd_binary != 0] = 1
    mask_pd_binary = RemoveSmallConnectedCompont2d(mask_pd_binary)
    mask_pd[mask_pd_binary == 0] = 0
    json_data = convertmasktojson(mask_pd)
    Vnet2d2.clear_GPU_cache()
    return json_data


def main():
    # config device
    case_name = os.listdir(INPUT_DIR)[0]
    print(case_name)
    # load image as numpy array
    case_image = cv2.imread(os.path.join(INPUT_DIR, case_name), 0)
    json_data = inferenceMutilVnet2dtestjsonv2(case_image)
    case_tag = case_name.split('.')[0]
    json_file = os.path.join(OUTPUT_DIR, '%s_Mask.json' % case_tag)
    with open(json_file, 'w') as json_file:
        json.dump(json_data, json_file, indent=4)


if __name__ == "__main__":
    main()
