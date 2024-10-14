# MICCAI STS 2024: Dental CBCT Scans & Panoramic X-ray Images
> This is an example of the CBCT and x-ray image is used to tooth segmentation.

![](cbct_logo.JPG)
![](x-ray_logo.JPG)

## Prerequisities
The following dependencies are needed:
- arrow==1.2.3
- binaryornot==0.4.4
- build==0.10.0
- certifi==2022.12.7
- chardet==5.1.0
- charset-normalizer==3.1.0
- click==8.1.3
- cookiecutter==2.1.1
- idna==3.4
- imageio[tifffile]==2.27.0
- jinja2==3.1.2
- jinja2-time==0.2.0
- joblib==1.2.0
- markupsafe==2.1.2
- numpy==1.21.6
- packaging==23.1
- pandas==1.3.5
- pillow==9.5.0
- pip-tools==6.13.0
- pyproject-hooks==1.0.0
- python-dateutil==2.8.2
- python-slugify==8.0.1
- pytz==2023.3
- pyyaml==6.0
- requests==2.28.2
- scikit-learn==1.0.2
- scipy==1.7.3
- simpleitk==2.2.1
- six==1.16.0
- text-unidecode==1.3
- threadpoolctl==3.1.0
- tifffile==2021.11.2
- tomli==2.0.1
- tzdata==2023.3
- urllib3==1.26.15
- wheel==0.40.0
- scikit-image==0.19.3
- evalutils==0.3.1
- opencv-python==4.7.0.68
- matplotlib==3.5.3
- torchsummary==1.5.1
- tensorboard==2.11.2
- onnx==1.13.0
- openslide-python==1.2.0
- pyvips==2.2.3
- seaborn>=0.11.0
- tqdm>=4.64.0
- PyYAML>=5.3.1
- setuptools>=65.5.1 # Snyk vulnerability fix
- thop>=0.1.1  # FLOPs computation
- torchinfo==1.8.0

## How to Use
* 1、download the whole project,install Python Environment using requirements.txt,and zip BinaryVNet3d.7z and MutilVNet3dModel.7z model files.
* 2、run task1 folder run_inference.py for x-ray tooth segmentation inference:make sure INPUT_DIR and OUTPUT_DIR has effective path.
* 3、run task2 folder run_inference.py for CBCT tooth segmentation inference:make sure INPUT_DIR and OUTPUT_DIR has effective path.

## Result

#  x-ray tooth segmentation predict result
![](task1_leadboard.JPG)
![](2d_图片1.png)
![](2d_图片1mask.png)
#  CBCT tooth segmentation predict result
![](task2_leadboard.JPG)
![](3d_图片1.png)
![](3d_图片2.png)

* you can find the x-ray and CBCT inference code in task1 and task2 folder.
* more detail and trained model can follow my WeChat Public article.

## Contact
* https://github.com/junqiangchen
* email: 1207173174@qq.com
* Contact: junqiangChen
* WeChat Number: 1207173174
* WeChat Public number: 最新医学影像技术
