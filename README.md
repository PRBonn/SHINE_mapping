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
  <h3 align="center"><a href="https://arxiv.org/abs/2210.02299">Arxiv</a> | <a href="https://www.ipb.uni-bonn.de/wp-content/papercite-data/pdf/zhong2023icra.pdf">Paper</a> | <a href="https://youtu.be/jRqIupJgQZE">Video</a></h3>
  <div align="center"></div>
</p>

<p align="center">
  <img src="https://user-images.githubusercontent.com/34207278/235914656-c56cd71f-1b31-44f7-a60d-3de4d77d76b6.png" width="100%" />
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
      <a href="#prepare-data">Prepare data</a>
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

Kaolin depends on Pytorch (>= 1.8, <= 1.13.1), please install the corresponding Pytorch for your CUDA version (can be checked by ```nvcc --version```). You can find the installation commands [here](https://pytorch.org/get-started/previous-versions/).

For example, for CUDA version >=11.6, you can use:
```
pip install torch==1.12.1+cu116 torchvision==0.13.1+cu116 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu116
```

Kaolin now supports installation with wheels. For example, to install kaolin 0.13.0 over torch 1.12.1 and cuda 11.6:
```
pip install kaolin==0.13.0 -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-1.12.1_cu116.html
```

<details>
  <summary>[Or you can build kaolin by yourself (click to expand)]</summary>

Follow the [instructions](https://kaolin.readthedocs.io/en/latest/notes/installation.html) to install [kaolin](https://kaolin.readthedocs.io/en/latest/index.html). Firstly, clone kaolin to local directory:

```
git clone --recursive https://github.com/NVIDIAGameWorks/kaolin
cd kaolin
```

Then install kaolin by:
```
python setup.py develop
```

Use ```python -c "import kaolin; print(kaolin.__version__)"``` to check if kaolin is successfully installed.
</details>


### 4. Install the other requirements
```
pip install open3d scikit-image wandb tqdm natsort 
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

The results will be stored with your experiment name with the starting timestamp in the `output_root` directory as what you set in the config file. You can find the reconstructed mesh (`*.ply` format) and optimized model in `mesh` and `model` folder, respectively. If the `save_map` option is truned on, then you can find the grid sdf map in `map` folder.

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

An interactive visualizer would pop up. You can press `space` to pause and resume.

<details>
  <summary>[Expected results (click to expand)]</summary>

For the sake of efficiency, we sacrifice a bit mapping quality to use a 50cm leaf voxel size for the feature octree.

<p align="center">
  <video src="https://user-images.githubusercontent.com/34207278/207639680-1060d60f-8ef1-4908-8d1f-9303f9020d4d.mp4" width="50%" />
</p>

</details>

For incremental mapping with replay strategy (within a local bounding box), use:
```
python shine_incre.py ./config/maicity/maicity_incre_replay.yaml
```

An interactive visualizer would pop up if you set `o3d_vis_on: True` (by default) in the config file. You can press `space` to pause and resume. 

<details>
  <summary>[Expected results (click to expand)]</summary>

For the sake of efficiency, we sacrifice a bit mapping quality to use a 50cm leaf voxel size for the feature octree here.

<p align="center">
  <video src="https://user-images.githubusercontent.com/34207278/207639846-4f80e55c-7574-45a5-9987-9c7ec6eb20e5.mp4" width="50%" />
</p>

</details>

<details>
  <summary>[Expected results on other datasets (click to expand)]</summary>

**KITTI**

<p align="center">
  <img src="https://user-images.githubusercontent.com/34207278/216335116-45273ef9-adb8-4f03-9e58-c5ca60b03081.png" width="70%" />
</p>

<p align="center">
  <img src="https://user-images.githubusercontent.com/34207278/235924403-1d1157f5-26c4-443e-b3dc-c87ee13c3e61.gif" width="70%" />
</p>

**Newer College**

<p align="center">
  <img src="https://user-images.githubusercontent.com/34207278/235925883-55a8230e-69b6-4ce0-a6c7-1f143de74504.gif" width="70%" />
</p>

**Apollo**

<p align="center">
  <img src="https://user-images.githubusercontent.com/34207278/235922691-d4e8cc0e-b85f-4542-9eaf-821bf26cc966.png" width="70%" />
</p>


**Wild Place Forests**

<p align="center">
  <img src="https://user-images.githubusercontent.com/34207278/207911217-c27a52c9-7233-4db9-a1fd-3487e59e6529.png" width="70%" />
</p>


**IPB Office**

<p align="center">
  <img src="https://user-images.githubusercontent.com/34207278/216360654-9b0a8bda-6a98-4db1-aa25-1c58080a4585.png" width="70%" />
</p>

**Replica**

<p align="center">
  <img src="https://user-images.githubusercontent.com/34207278/235925819-bc30934f-8df0-465c-b157-d2702022d06e.gif" width="70%" />
</p>

**ICL Living Room**

<p align="center">
  <img src="https://user-images.githubusercontent.com/34207278/235922527-a39f6e1c-443b-462e-842d-f33e0e5716be.png" width="70%" />
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

<details>
  <summary>[Details (click to expand)]</summary>

1. You can play with different loss functions for SHINE Mapping. With the `ray_loss: False` option, the loss would be calculated from the sdf at each sample point. In this case, you can then select from `sdf_bce` (the proposed method), `sdf_l1` and  `sdf_l2` loss as the `main_loss_type`. With the `ray_loss: True` option, the loss would be calculated from each ray conatining multiple point samples as a depth rendering procedure. In this case,  you can select from `dr` and `dr_neus` as the `main_loss_type`. According to our experiments, using our proposed `sdf_bce` loss can achieve the best reconstruction efficiently. We can get a decent reconstruction of a scene with several hundred frames in just one minute. Additionally, you can use the `ekional_loss_on` option to turn on/off the Ekional loss and use `weight_e` as its weight.

2. The feature octree is built mainly according to `leaf_vox_size`, `tree_level_world` and `tree_level_feat`. `leaf_vox_size` represents the size of the leaf voxel size in meter. `tree_level_world` and `tree_level_feat` represent the total tree level and the tree levels with latent feature codes, respectively. `tree_level_world` should be large enough to gurantee all the map data lies inside the cube with the size `leaf_vox_size**(tree_level_world+1)`.

3. SHINE Mapping supports both the offline batch mapping and the incremental sequential mapping. For incremental mapping, one can either load a fixed pre-trained decoder from the batching mapping on a similar dataset (set `load_model: True`) or train the decoder for `freeze_after_frame` frames on-the-fly and then freeze it afterwards (set `load_model: False`). The first option would lead to better mapping performance.

4. You can use the `mc_vis_level` parameter to have a trade-off between the scene completion and the exact measurement accuracy. This parameter indicate at which level of the octree the marching cubes reconstruction would be conducted. The larger the value of `mc_vis_level` (but not larger than `tree_level_feat`), the more scene completion ability you would gain (but also some artifacts such as a double wall may appear). And with the small value, SHINE mapping would only reconstruct the part with actual measurements without filling the holes. The safest way to avoid the holes on the ground is to set `mc_mask_on: False` to disable the masking for marching cubes. By turning on the `mc_with_octree` option, you can achieve a faster marching cubes reconstruction only in the region within the octree nodes. 

5. The incremental mapping with regularization strategy (setting `continual_learning_reg: True`) can achieve incremental neural mapping without storing an ever-growing data pool which would be a burden for the memory. The coefficient `lambda_forget` needs to be fine-tuned under different feature octree and point sampling settings. The recommended value is from `1e5` to `1e8`. A pre-trained decoder is also recommended to be loaded during incremental mapping with regularization for better performance. 

6. We also provide an option to conduct incremental mapping with replay strategy in a local sliding window. You can turn this on by setting `window_replay_on: True` with a valid `window_radius_m` indicating the size of the sliding window.

7. It's also possible to incoporate semantic information in our SHINE-Mapping framework. You may set `semantic_on = True` in the `utils/config.py` file to enable semantic mapping and also provide the semantic supervision by setting the `label_path` in the config file.

</details>

----

## Citation
If you use SHINE Mapping for any academic work, please cite our [original paper](https://www.ipb.uni-bonn.de/wp-content/papercite-data/pdf/zhong2023icra.pdf).
```
@inproceedings{zhong2023icra,
  title={SHINE-Mapping: Large-Scale 3D Mapping Using Sparse Hierarchical Implicit NEural Representations},
  author={Zhong, Xingguang and Pan, Yue and Behley, Jens and Stachniss, Cyrill},
  booktitle = {Proceedings of the IEEE International Conference on Robotics and Automation (ICRA)},
  year={2023}
}
```

## Contact
If you have any questions, please contact:

- Xingguang Zhong {[zhong@igg.uni-bonn.de]()}
- Yue Pan {[yue.pan@igg.uni-bonn.de]()}

## Acknowledgment
This work has partially been funded by the European Union’s HORIZON programme under grant agreement No 101070405 (DigiForest) and grant agreement No 101017008 (Harmony).

Additional, we thank greatly for the authors of the following opensource projects:

- [NGLOD](https://github.com/nv-tlabs/nglod) (octree based hierarchical feature structure built based on [kaolin](https://kaolin.readthedocs.io/en/latest/index.html)) 
- [VDBFusion](https://github.com/PRBonn/vdbfusion) (comparison baseline)
- [Voxblox](https://github.com/ethz-asl/voxblox) (comparison baseline)
- [Puma](https://github.com/PRBonn/puma) (comparison baseline and the MaiCity dataset)
- [KISS-ICP](https://github.com/PRBonn/kiss-icp) (simple yet effective pose estimation)

