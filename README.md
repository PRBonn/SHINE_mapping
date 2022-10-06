# ✨ SHINE-Mapping: Large-Scale 3D Mapping Using Sparse Hierarchical Implicit Neural Representations
This repository contains the implementation of the paper:

SHINE-Mapping: Large-Scale 3D Mapping Using **S**parse **H**ierarchical **I**mplicit **NE**ural Representations.

Xingguang Zhong, Yue Pan, Jens Behley and Cyrill Stachniss

[Link](https://arxiv.org/abs/2210.02299) to the arXiv version of the paper.

----
![teaser_for_github](https://user-images.githubusercontent.com/34207278/192101055-271aafe2-7198-46b2-ae70-dae8d3bc0f97.png)

## Demo Video

Incremental Mapping | Reconstruction Results |
:-: | :-: |
<video src='https://user-images.githubusercontent.com/34207278/192112474-f88d0d90-96a4-4ff3-b3bb-4e119b810d9e.mp4'> | <video src='https://user-images.githubusercontent.com/34207278/192112449-56cb5c73-500f-416a-8892-e44d0e962669.mp4'> |


## Abstract

Accurate mapping of large-scale environments is an essential building block of most outdoor autonomous systems. Challenges of traditional mapping methods include the balance between memory consumption and mapping accuracy. This paper addresses the problems of achieving large-scale 3D reconstructions with implicit representations using 3D LiDAR measurements. We learn and store implicit features through an octree-based hierarchical structure, which is sparse and extensible. The features can be turned into signed distance values through a shallow neural network. We leverage binary cross entropy loss to optimize the local features with the 3D measurements as supervision. Based on our implicit representation, we design an incremental mapping system with regularization to tackle the issue of catastrophic forgetting in continual learning. Our experiments show that our 3D reconstructions are more accurate, complete, and memory-efficient than current state-of-the-art 3D mapping methods.


## Codes
Coming soon.





