# ✨ SHINE-Mapping: Large-Scale 3D Mapping Using Sparse Hierarchical Implicit Neural Representations

This repository contains the implementation of the paper:

SHINE-Mapping: Large-Scale 3D Mapping Using **S**parse **H**ierarchical **I**mplicit **NE**ural Representations.

By [Xingguang Zhong](https://www.ipb.uni-bonn.de/people/xingguang-zhong/), [Yue Pan](https://www.ipb.uni-bonn.de/people/yue-pan/), [Jens Behley](https://www.ipb.uni-bonn.de/people/jens-behley/) and [Cyrill Stachniss](https://www.ipb.uni-bonn.de/people/cyrill-stachniss/)

[Arxiv Preprint](https://arxiv.org/abs/2210.02299) | [Demo Video]()

![teaser_fig](https://user-images.githubusercontent.com/34207278/194295874-ccf02ed0-ad10-4451-acd2-e70001737ecf.png)

## Demo Video

Incremental Mapping | Reconstruction Results |
:-: | :-: |
<video src='https://user-images.githubusercontent.com/34207278/192112474-f88d0d90-96a4-4ff3-b3bb-4e119b810d9e.mp4'> | <video src='https://user-images.githubusercontent.com/34207278/192112449-56cb5c73-500f-416a-8892-e44d0e962669.mp4'> |


## Abstract
Accurate mapping of large-scale environments is an essential building block of most outdoor autonomous systems. Challenges of traditional mapping methods include the balance between memory consumption and mapping accuracy. This paper addresses the problems of achieving large-scale 3D reconstructions with implicit representations using 3D LiDAR measurements. We learn and store implicit features through an octree-based hierarchical structure, which is sparse and extensible. The features can be turned into signed distance values through a shallow neural network. We leverage binary cross entropy loss to optimize the local features with the 3D measurements as supervision. Based on our implicit representation, we design an incremental mapping system with regularization to tackle the issue of catastrophic forgetting in continual learning. Our experiments show that our 3D reconstructions are more accurate, complete, and memory-efficient than current state-of-the-art 3D mapping methods.

## Installation

### 1. Clone SHINE Mapping repository
```
git clone git@github.com:PRBonn/SHINE_mapping.git
cd SHINE_mapping
```
### 2. Set up conda environment
```
conda create --name shine python=3.7
conda activate shine
```
### 3. Install the key requirement kaolin

Follow the [instructions](https://kaolin.readthedocs.io/en/latest/notes/installation.html) to install [kaolin](https://kaolin.readthedocs.io/en/latest/index.html). First clone kaolin to local directory:

```
git clone --recursive https://github.com/NVIDIAGameWorks/kaolin
cd kaolin
```

Kaolin depends on Pytorch (>= 1.8, <= 1.12.1). Please install the corresponding Pytorch for your CUDA version (can be checked by ```nvcc --version```). You can find the installation commands [here](https://pytorch.org/get-started/previous-versions/).

For example, for CUDA version >=11.6, you can use:
```
pip install torch==1.12.1+cu116 torchvision==0.13.1+cu116 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu116
```

Then install kaolin by:
```
python setup.py develop
```

Use ```python -c "import kaolin; print(kaolin.__version__)"``` to check if kaolin is successfully installed.

### 4. Install the other requirements
```
pip install open3d wandb
conda install scikit-image
```

## Prepare data

Generally speaking, we only need to provide:
1. the folder containing the point cloud (`.bin`, `.ply` or `.pcd` format) for each frame.
2. the pose file (`.txt`) containing the transformation matrix of each frame.
3. the calib file (`.txt`) containing the static transformation between sensor and body frames.

They all follow the [KITTI odometry data format](https://www.cvlibs.net/datasets/kitti/eval_odometry.php).

### MaiCity synthetic LiDAR dataset

Download the dataset from [here](https://www.ipb.uni-bonn.de/data/mai-city-dataset/)(3.4G) or use the following command in a target data folder:

```
wget https://www.ipb.uni-bonn.de/html/projects/mai_city/mai_city.tar.gz
tar -xvf mai_city.tar.gz
```

### KITTI real-world LiDAR dataset

Download the dataset from [here](https://www.cvlibs.net/datasets/kitti/eval_odometry.php)(80G).

### Newer College real-world LiDAR dataset

Download the dataset from [here]().


After downloading the data, you need to correctly set the data path (`pc_path`, `pose_path` and `calib_path`) in the config files under `config` folder.

## Run

For batch processing based mapping, use:
```
python shine_batch.py ./config/[dataset_name]/[dataset_name]_batch.yaml
```

For incremental mapping with regularization, use:
```
python shine_incre.py ./config/[dataset_name]/[dataset_name]_incre_reg.yaml
```

For incremental mapping with replay, use:
```
python shine_incre.py ./config/[dataset_name]/[dataset_name]_incre_replay.yaml
```

The results will be stored in the `output_root` directory as you set in the config file.
The logs can be monitored via Weight & Bias online if you turn the `wandb_vis_on` option on. 

## Evaluation

TBD

----

## Acknowledgment
This work has partially been funded by the European Union’s HORIZON programme under grant agreement No 101070405 (DigiForest).





