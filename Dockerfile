# Build: docker build --tag shine .
#
# Make sure you have the input directory (MY_DATA_DIR) prepared as can be seen in kitti/docker_kitti_config.yaml
#
# Run: docker run --rm -v .:/repository -v ${MY_DATA_DIR}:/data -it --gpus all shine bash
# Run: (with vis): xhost local:root && docker run --rm -v /tmp/.X11-unix:/tmp/.X11-unix -e DISPLAY -v .:/repository -v ${MY_DATA_DIR}:/data -it --gpus 'all,"capabilities=compute,utility,graphics"' shine

FROM nvidia/cuda:11.6.2-devel-ubuntu20.04

ARG FORCE_CUDA=1
ENV FORCE_CUDA=${FORCE_CUDA}

RUN apt-get update && apt-get install --no-install-recommends -y \
    python3 \
    python3-dev \
    python3-pip \
    wget \
    git \
    libgl1-mesa-glx \
    vim \
    && apt-get clean

RUN python3 -m pip install --upgrade pip
RUN pip3 install torch==1.12.1+cu116 torchvision==0.13.1+cu116 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu116
RUN pip3 install open3d==0.17.0 scikit-image wandb tqdm natsort pyquaternion

SHELL ["/bin/bash", "-c"]

WORKDIR /opt
RUN git clone --recursive https://github.com/NVIDIAGameWorks/kaolin

WORKDIR /opt/kaolin
RUN git checkout v0.13.0
RUN pip3 install -r ./tools/requirements.txt
RUN python3 setup.py develop 

WORKDIR /repository

CMD ["bash", "-c", ". ./scripts/download_kitti_example.sh && mv /repository/data/kitti_example/sequences/00/* /data/ && mkdir -p /data/results && cd /repository && python3 shine_batch.py ./config/kitti/docker_kitti_batch.yaml"]
