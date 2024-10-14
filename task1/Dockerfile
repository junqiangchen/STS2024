ARG PYTORCH="1.12.1"
ARG CUDA="11.3"
ARG CUDNN="8"

FROM python:3.8.3

# Pull the docker image | 拉取镜像
# FROM pytorch/pytorch
FROM pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-devel
# only work on linux system
#FROM nvidia/cuda:11.4.3-cudnn8-devel-ubuntu18.04

# 安装所需的软件包
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    wget \
    unzip \
    yasm \
    pkg-config \
    libswscale-dev \
    libtbb2 \
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    libavformat-dev \
    libpq-dev \
    python3-pip \
    libgl1-mesa-dev \
    python-openslide \
    libvips-dev


ENV PYTHONUNBUFFERED 1

RUN groupadd -r user && useradd -m --no-log-init -r -g user user
USER user

WORKDIR /opt/app

# Add the directory containing the scripts to PATH
ENV PATH="/home/user/.local/bin:$PATH"
ENV LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH}"

## Copy all files
COPY --chown=user:user ./ /opt/app/
COPY --chown=user:user requirements.txt /opt/app/

## Install Python packages in Docker image
RUN pip3 install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

RUN python -m pip install \
    --user \
    --no-cache-dir \
    --no-color \
    --requirement /opt/app/requirements.txt

COPY --chown=user:user run_inference.py /opt/app/

ENTRYPOINT ["python", "run_inference.py"]
