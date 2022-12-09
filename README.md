<p align="center">

  <h1 align="center">✨ SHINE-Mapping: Large-Scale 3D Mapping Using Sparse Hierarchical Implicit Neural Representations</h1>
  <p align="center">
    <a href="https://www.ipb.uni-bonn.de/people/xingguang-zhong/"><strong>Xingguang Zhong*</strong></a>
    ·
    <a href="https://www.ipb.uni-bonn.de/people/yue-pan/"><strong>Yue Pan*</strong></a>
    ·
    <a href="https://www.ipb.uni-bonn.de/people/jens-behley/"><strong>Jens Behley</strong></a>
    ·
    <a href="https://www.ipb.uni-bonn.de/people/cyrill-stachniss/"><strong>Cyrill Stachniss</strong></a>
  </p>
  <p align="center"><a href="https://www.ipb.uni-bonn.de"><strong>University of Bonn</strong></a>
  <p align="center">(* Equal Contribution)</p>
  <h3 align="center"><a href="https://arxiv.org/abs/2210.02299">Arxiv Preprint</a> | <a href="">Video</a></h3>
  <div align="center"></div>
</p>


Incremental Mapping | Reconstruction Results |
:-: | :-: |
<video src='https://user-images.githubusercontent.com/34207278/192112474-f88d0d90-96a4-4ff3-b3bb-4e119b810d9e.mp4'> | <video src='https://user-images.githubusercontent.com/34207278/192112449-56cb5c73-500f-416a-8892-e44d0e962669.mp4'> |


<!-- TABLE OF CONTENTS -->
<details open="open" style='padding: 10px; border-radius:5px 30px 30px 5px; border-style: solid; border-width: 1px;'>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#abstract">Abstract</a>
    </li>
    <li>
      <a href="#installation">Installation</a>
    </li>
    <li>
      <a href="#prepare data">Prepare data</a>
    </li>
    <li>
      <a href="#run">How to run</a>
    </li>
    <li>
      <a href="#evaluation">Evaluation</a>
    </li>
    <li>
      <a href="#tips">Tips</a>
    </li>
    <li>
      <a href="#citation">Citation</a>
    </li>
    <li>
      <a href="#contact">Contact</a>
    </li>
    <li>
      <a href="#acknowledgment">Acknowledgment</a>
    </li>
  </ol>
</details>

----
## Abstract
Accurate mapping of large-scale environments is an essential building block of most outdoor autonomous systems. Challenges of traditional mapping methods include the balance between memory consumption and mapping accuracy. This paper addresses the problems of achieving large-scale 3D reconstructions with implicit representations using 3D LiDAR measurements. We learn and store implicit features through an octree-based hierarchical structure, which is sparse and extensible. The features can be turned into signed distance values through a shallow neural network. We leverage binary cross entropy loss to optimize the local features with the 3D measurements as supervision. Based on our implicit representation, we design an incremental mapping system with regularization to tackle the issue of catastrophic forgetting in continual learning. Our experiments show that our 3D reconstructions are more accurate, complete, and memory-efficient than current state-of-the-art 3D mapping methods.

----
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

Follow the [instructions](https://kaolin.readthedocs.io/en/latest/notes/installation.html) to install [kaolin](https://kaolin.readthedocs.io/en/latest/index.html). Firstly, clone kaolin to local directory:

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
pip install open3d wandb tqdm
conda install scikit-image
```

----

## Prepare data

Generally speaking, you only need to provide:
1. `pc_path` : the folder containing the point cloud (`.bin`, `.ply` or `.pcd` format) for each frame.
2. `pose_path` : the pose file (`.txt`) containing the transformation matrix of each frame. 
3. `calib_path` : the calib file (`.txt`) containing the static transformation between sensor and body frames (optional, would be identity matrix if set as `''`).

They all follow the [KITTI odometry data format](https://www.cvlibs.net/datasets/kitti/eval_odometry.php).

After preparing the data, you need to correctly set the data path (`pc_path`, `pose_path` and `calib_path`) in the config files under `config` folder. You may also set a path `output_root` to store the experiment results and logs.

Here, we provide the link to several public available datasets for testing SHINE Mapping:

### MaiCity synthetic LiDAR dataset

Download the dataset from [here](https://www.ipb.uni-bonn.de/data/mai-city-dataset/) or use the following script to download (3.4GB):

```
sh ./scripts/download_maicity.sh
```

### KITTI real-world LiDAR dataset

Download the full dataset from [here](https://www.cvlibs.net/datasets/kitti/eval_odometry.php).

If you want to use an example part of the dataset (seq 00) for the test, you can use the following script to download (117 MB):
```
sh ./scripts/download_kitti_example.sh
```

### Newer College real-world LiDAR dataset

Download the full dataset from [here](https://ori-drs.github.io/newer-college-dataset/download/).

If you want to use an example part of the dataset (Quad) for the test, you can use the following script to download (634 MB):
```
sh ./scripts/download_ncd_example.sh
```

### RGB-D datasets

SHINE Mapping also supports the mapping on RGB-D datasets. You may firstly try the synthetic dataset from [NeuralRGB-D](https://github.com/dazinovic/neural-rgbd-surface-reconstruction). You can download the full dataset from [here](http://kaldir.vc.in.tum.de/neural_rgbd/neural_rgbd_data.zip) or use the following script to download (7.25 GB).
```
sh ./scripts/download_neural_rgbd_data.sh
```

After downloading the data, you need to convert the dataset to the KITTI format by using for each sequence:
```
sh ./scripts/convert_rgbd_to_kitti_format.sh
```

### Mapping without ground truth pose
<details>
  <summary>[Details (click to expand)]</summary>

Our method is currently a mapping-with-known-pose system. If you do not have the ground truth pose file, you may use a LiDAR odometry system such as [KISS-ICP](https://github.com/PRBonn/kiss-icp) to easily estimate the pose.  

You can simply install KISS-ICP by:

```
pip install kiss-icp
```
And then run KISS-ICP with your data path `pc_path`
```
kiss_icp_pipeline <pc_path>
```
The estimated pose file can be find in `./results/latest/velodyne.txt`. You can directly use it as your `pose_path`. In this case, you do not need a calib file, so just set `calib_path: ""` in the config file.
</details>

----

## Run

We take the MaiCity dataset as an example to show how SHINE Mapping works. You can simply replace maicity with your dataset name in the config file path, such as `./config/[dataset]/[dataset]_[xxx].yaml`.

The results will be stored with your experiment name with the starting timestamp in the `output_root` directory as what you set in the config file. You can find the reconstructed mesh (`*.ply` format) and optimized model in `mesh` and `model` folder, respectively.


For mapping based on offline batch processing, use:
```
python shine_batch.py ./config/maicity/maicity_batch.yaml
```

<details>
  <summary>[Expected results (click to expand)]</summary>

<p align="center">
  <video src="https://user-images.githubusercontent.com/34207278/206579093-8ba92baa-2b98-462a-b92d-ce3eff8ede64.mp4" width="50%" />
</p>

</details>


For incremental mapping with regularization strategy, use:
```
python shine_incre.py ./config/maicity/maicity_incre_reg.yaml
```

<details>
  <summary>[Expected results (click to expand)]</summary>

For the sake of efficiency, we use a 50cm leaf voxel size for the feature octree.

<p align="center">
  <video src="https://user-images.githubusercontent.com/34207278/206582367-f2f4a66a-83f8-4a90-a2db-e104b5985bdd.mp4" width="50%" />
</p>

</details>

For incremental mapping with replay strategy, use:
```
python shine_incre.py ./config/maicity/maicity_incre_replay.yaml
```
<details>
  <summary>[Expected results (click to expand)]</summary>

For the sake of efficiency, we use a 50cm leaf voxel size for the feature octree.

<p align="center">
  <video src="https://user-images.githubusercontent.com/34207278/206581857-0b38bf26-ef00-4983-a903-4a58ae7ff001.mp4" width="50%" />
</p>

</details>

<details>
  <summary>[Expected results on other datasets (click to expand)]</summary>

**KITTI**

<p align="center">
  <img src="https://user-images.githubusercontent.com/34207278/206718854-aefa02ec-72ac-470e-a3d0-d61e7d886692.png" width="70%" />
</p>

**Newer College**

<p align="center">
  <img src="https://user-images.githubusercontent.com/34207278/206719372-477f1a90-e2b1-44ac-a56e-22e8e5810147.png" width="70%" />
</p>!

**RGB-D**

<p align="center">
  <img src="https://user-images.githubusercontent.com/34207278/206718688-a8bfb786-6439-435b-a225-c596aef9ab64.png" width="70%" />
</p>


</details>

The logs can be monitored via [Weights & Bias](https://wandb.ai/site) online if you turn the `wandb_vis_on` option on. If it's your first time to use Weights & Bias, you would be requested to register and login to your wandb account. 

## Evaluation

To evaluate the reconstruction quality, you need to provide the (reference) ground truth point cloud and your reconstructed mesh. The ground truth point cloud can be found (or sampled from) the downloaded folder of MaiCity, Newer College and Neural RGBD datasets. 

Please change the data path and evaluation set-up in `./eval/evaluator.py` and then run:

```
python ./eval/evaluator.py
```

to get the reconstruction metrics such as Chamfer distance, completeness, F-score, etc.

As mentioned in the paper, we also compute a fairer accuracy metric using the ground truth point cloud masked by the intersection of the reconstructed meshes of all the compared methods. To generate such masked ground truth point clouds, you can configure the data path in `./eval/crop_intersection.py` and then run it.

## Tips

1. You can play with different loss functions for SHINE Mapping. With the `ray_loss: False` option, the loss would be calculated from the sdf at each sample point. In this case, you can then select from `sdf_bce` (the proposed method), `sdf_l1` and  `sdf_l2` loss as the `main_loss_type`. With the `ray_loss: True` option, the loss would be calculated from each ray conatining multiple point samples as a depth rendering procedure. In this case,  you can select from `dr` and `dr_neus` as the `main_loss_type`. Additionally, you can set `ekional_loss_on` option to turn on/off the Ekional loss and use `weight_e` as its weight.

2. The feature octree is built mainly according to `leaf_vox_size`, `tree_level_world` and `tree_level_feat`. `leaf_vox_size` represents the size of the leaf voxel size in meter. `tree_level_world` and `tree_level_feat` represent the total tree level and the tree levels with latent feature codes, respectively. `tree_level_world` should be large enough to gurantee all the map data lies inside the cube with the size `leaf_vox_size**(tree_level_world+1)`.

----

## Citation
If you use SHINE Mapping for any academic work, please cite our original paper.
```
@article{zhong2022arxiv,
  title={SHINE-Mapping: Large-Scale 3D Mapping Using Sparse Hierarchical Implicit NEural Representations},
  author={Zhong, Xingguang and Pan, Yue and Behley, Jens and Stachniss, Cyrill},
  journal={arXiv preprint arXiv:2210.02299},
  year={2022}
}
```

## Contact
If you have any questions, please contact:

- Xingguang Zhong {[zhong@igg.uni-bonn.de]()}
- Yue Pan {[yue.pan@igg.uni-bonn.de]()}

## Acknowledgment
This work has partially been funded by the European Union’s HORIZON programme under grant agreement No 101070405 (DigiForest).

Additional, we thanks greatly for the authors of the following opensource projects:

- [NGLOD](https://github.com/nv-tlabs/nglod) (octree based hierarchical feature structure built based on [kaolin](https://kaolin.readthedocs.io/en/latest/index.html)) 
- [VDBFusion](https://github.com/PRBonn/vdbfusion) (comparison baseline)
- [Voxblox](https://github.com/ethz-asl/voxblox) (comparison baseline)
- [Puma](https://github.com/PRBonn/puma) (comparison baseline and the MaiCity dataset)
- [KISS-ICP](https://github.com/PRBonn/kiss-icp) (simple yet effective pose estimation)

