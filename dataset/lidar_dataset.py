import os
import sys
import numpy as np
from numpy.linalg import inv, norm
from tqdm import tqdm
import copy
import torch
from torch.utils.data import Dataset
import open3d as o3d

from utils.config import SHINEConfig
from utils.tools import get_time
from utils.pose import read_calib_file, read_poses_file
from utils.data_sampler import dataSampler
from utils.semantic_kitti_utils import *
from model.feature_octree import FeatureOctree


class LiDARDataset(Dataset):
    def __init__(self, config: SHINEConfig, octree: FeatureOctree) -> None:

        super().__init__()

        self.config = config
        self.dtype = config.dtype
        torch.set_default_dtype(self.dtype)

        self.calib = {}
        if config.calib_path != '':
            self.calib = read_calib_file(config.calib_path)
        else:
            self.calib['Tr'] = np.eye(4)
        self.poses_w = read_poses_file(config.pose_path, self.calib)

        # pose in the reference frame (might be the first frame used)
        self.poses_ref = self.poses_w  # initialize size

        # point cloud files
        self.pc_filenames = sorted(os.listdir(config.pc_path))
        self.total_pc_count = len(self.pc_filenames)

        dev = config.device
        self.coord_pool = torch.empty((0, 3), device=dev, dtype=self.dtype)
        self.sdf_label_pool = torch.empty((0), device=dev, dtype=self.dtype)
        self.normal_label_pool = torch.empty((0, 3), device=dev, dtype=self.dtype)
        self.sem_label_pool = torch.empty((0), device=dev, dtype=int)
        self.weight_pool = torch.empty((0), device=dev, dtype=self.dtype)
        self.sample_depth_pool = torch.empty((0), device=dev, dtype=self.dtype)
        self.ray_depth_pool = torch.empty((0), device=dev, dtype=self.dtype)

        # feature octree
        self.octree = octree

        # initialize the data sampler
        self.sampler = dataSampler(config)
        self.ray_sample_count = config.surface_sample_n + config.free_sample_n

        # merged downsampled point cloud
        self.map_down_pc = o3d.geometry.PointCloud()
        # map bounding box in the world coordinate system
        self.map_bbx = o3d.geometry.AxisAlignedBoundingBox()

        # get the pose in the reference frame
        if config.first_frame_ref:
            begin_flag = False
            begin_pose_inv = np.eye(4)
            for frame_id in range(self.total_pc_count):
                if (
                    frame_id < config.begin_frame
                    or frame_id > config.end_frame
                    or frame_id % config.every_frame != 0
                ):
                    continue
                if not begin_flag:  # the first frame used
                    begin_flag = True
                    begin_pose_inv = inv(self.poses_w[frame_id])  # T_rw
                # use the first frame as the reference (identity)
                self.poses_ref[frame_id] = np.matmul(
                    begin_pose_inv, self.poses_w[frame_id]
                )
        # or we directly use the world frame as reference

    def process_frame(self, frame_id, incremental_on = False):

        dev = self.config.device
        # dev = 'cpu'
        pc_radius = self.config.pc_radius
        min_z = self.config.min_z
        max_z = self.config.max_z
        normal_radius_m = self.config.normal_radius_m
        normal_max_nn = self.config.normal_max_nn
        rand_down_r = self.config.rand_down_r
        vox_down_m = self.config.vox_down_m
        sor_nn = self.config.sor_nn
        sor_std = self.config.sor_std

        self.cur_pose_ref = self.poses_ref[frame_id]
        frame_origin = self.cur_pose_ref[:3, 3] * self.config.scale  # translation part
        frame_origin_torch = torch.tensor(frame_origin, dtype=self.dtype, device=dev)

        # load point cloud (support *pcd, *ply and kitti *bin format)
        frame_filename = os.path.join(self.config.pc_path, self.pc_filenames[frame_id])
        
        if not self.config.semantic_on:
            frame_pc = self.read_point_cloud(frame_filename)
        else:
            label_filename = os.path.join(self.config.label_path, self.pc_filenames[frame_id].replace('bin','label'))
            frame_pc = self.read_semantic_point_label(frame_filename, label_filename)

        # block filter: crop the point clouds into a cube
        bbx_min = np.array([-pc_radius, -pc_radius, min_z])
        bbx_max = np.array([pc_radius, pc_radius, max_z])
        bbx = o3d.geometry.AxisAlignedBoundingBox(bbx_min, bbx_max)
        frame_pc = frame_pc.crop(bbx)

        # surface normal estimation
        if self.config.estimate_normal:
            frame_pc.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamHybrid(
                    radius=normal_radius_m, max_nn=normal_max_nn
                )
            )

        # point cloud downsampling
        if self.config.rand_downsample:
            # random downsampling
            frame_pc = frame_pc.random_down_sample(sampling_ratio=rand_down_r)
        else:
            # voxel downsampling
            frame_pc = frame_pc.voxel_down_sample(voxel_size=vox_down_m)

        # apply filter (optional)
        if self.config.filter_noise:
            frame_pc = frame_pc.remove_statistical_outlier(
                sor_nn, sor_std, print_progress=False
            )[0]

        # load the label from the color channel of frame_pc
        if self.config.semantic_on:
            frame_sem_label = np.asarray(frame_pc.colors)[:,0]*255.0 # from [0-1] tp [0-255]
            frame_sem_label = np.round(frame_sem_label, 0) # to integer value
            sem_label_list = list(frame_sem_label)
            frame_sem_rgb = [sem_kitti_color_map[sem_label] for sem_label in sem_label_list]
            frame_sem_rgb = np.asarray(frame_sem_rgb)/255.0
            frame_pc.colors = o3d.utility.Vector3dVector(frame_sem_rgb)

        # transform to reference frame 
        frame_pc = frame_pc.transform(self.cur_pose_ref)
        # make a backup for merging into the map point cloud
        frame_pc_clone = copy.deepcopy(frame_pc)
        frame_pc_clone = frame_pc_clone.voxel_down_sample(voxel_size=self.config.map_vox_down_m) # for smaller memory cost
        self.map_down_pc += frame_pc_clone
        self.cur_frame_pc = frame_pc_clone

        self.map_bbx = self.map_down_pc.get_axis_aligned_bounding_box()
        # and scale to [-1,1] coordinate system
        frame_pc_s = frame_pc.scale(self.config.scale, center=(0,0,0))

        frame_pc_s_torch = torch.tensor(np.asarray(frame_pc_s.points), dtype=self.dtype, device=dev)

        frame_normal_torch = None
        if self.config.estimate_normal:
            frame_normal_torch = torch.tensor(np.asarray(frame_pc_s.normals), dtype=self.dtype, device=dev)

        frame_label_torch = None
        if self.config.semantic_on:
            frame_label_torch = torch.tensor(frame_sem_label, dtype=self.dtype, device=dev)

        # print("Frame point cloud count:", frame_pc_s_torch.shape[0])

        # sampling the points
        (coord, sdf_label, normal_label, sem_label, weight, sample_depth, ray_depth) = \
            self.sampler.sample(frame_pc_s_torch, frame_origin_torch, \
            frame_normal_torch, frame_label_torch)

        # ray-wise samples order
        if incremental_on:
            self.coord_pool = coord
            self.sdf_label_pool = sdf_label
            self.normal_label_pool = normal_label
            self.sem_label_pool = sem_label
            self.weight_pool = weight
            self.sample_depth_pool = sample_depth
            self.ray_depth_pool = ray_depth        
        else: # batch processing
            self.coord_pool = torch.cat((self.coord_pool, coord), 0)            
            self.weight_pool = torch.cat((self.weight_pool, weight), 0)
            if self.config.ray_loss:
                self.sample_depth_pool = torch.cat((self.sample_depth_pool, sample_depth), 0)
                self.ray_depth_pool = torch.cat((self.ray_depth_pool, ray_depth), 0)
            else:
                self.sdf_label_pool = torch.cat((self.sdf_label_pool, sdf_label), 0)

            if normal_label is not None:
                self.normal_label_pool = torch.cat((self.normal_label_pool, normal_label), 0)
            else:
                self.normal_label_pool = None
            
            if sem_label is not None:
                self.sem_label_pool = torch.cat((self.sem_label_pool, sem_label), 0)
            else:
                self.sem_label_pool = None

        # update feature octree
        if self.config.octree_from_surface_samples:
            # update with the sampled surface points
            self.octree.update(coord[weight > 0, :], incremental_on)
        else:
            # update with the original points
            self.octree.update(frame_pc_s_torch.to("cuda"), incremental_on)  

    def read_point_cloud(self, filename: str):
        # read point cloud from either (*.ply, *.pcd) or (kitti *.bin) format
        if ".bin" in filename:
            points = np.fromfile(filename, dtype=np.float32).reshape((-1, 4))[:, :3]
        elif ".ply" in filename or ".pcd" in filename:
            pc_load = o3d.io.read_point_cloud(filename)
            points = np.asarray(pc_load.points)
        else:
            sys.exit(
                "The format of the imported point cloud is wrong (support only *pcd, *ply and *bin)"
            )
        preprocessed_points = self.preprocess_kitti(
            points, self.config.min_z, self.config.min_range
        )
        pc_out = o3d.geometry.PointCloud()
        pc_out.points = o3d.utility.Vector3dVector(preprocessed_points)
        return pc_out

    def read_semantic_point_label(self, bin_filename: str, label_filename: str):

        # read point cloud (kitti *.bin format)
        if ".bin" in bin_filename:
            points = np.fromfile(bin_filename, dtype=np.float32).reshape((-1, 4))[:, :3]
        else:
            sys.exit(
                "The format of the imported point cloud is wrong (support only *bin)"
            )

        # read point cloud labels (*.label format)
        if ".label" in label_filename:
            labels = np.fromfile(label_filename, dtype=np.uint32).reshape((-1))
        else:
            sys.exit(
                "The format of the imported point labels is wrong (support only *label)"
            )

        points, labels = self.preprocess_sem_kitti(
            points, labels, self.config.min_z, self.config.min_range
        )

        sem_labels = labels & 0xFFFF
        # ins_labels = labels >> 16

        sem_labels_coarse = [sem_kitti_learning_map[sem_label] for sem_label in sem_labels]

        # sem_rgb = [sem_kitti_color_map[sem_label_coarse] for sem_label_coarse in sem_labels_coarse]

        sem_labels_coarse = (np.asarray(sem_labels_coarse)/255.0).reshape((-1, 1)).repeat(3, axis=1) # label 

        # better to use o3d.t.geometry.PointCloud(device)
        # then you can use sdf_map_pc.point['positions'], sdf_map_pc.point['intensities'], sdf_map_pc.point['labels']
        pc_out = o3d.geometry.PointCloud()
        pc_out.points = o3d.utility.Vector3dVector(points)
        pc_out.colors = o3d.utility.Vector3dVector(sem_labels_coarse)

        return pc_out

    def preprocess_kitti(self, points, z_th=-3.0, min_range=2.5):
        # filter the outliers
        z = points[:, 2]
        points = points[z > z_th]
        points = points[np.linalg.norm(points, axis=1) >= min_range]
        return points

    def preprocess_sem_kitti(self, points, labels, z_th=-3.0, min_range=2.5, filter_moving = True):
        # filter the outliers
        
        # z = points[:, 2]
        # points = points[z > z_th]
        # labels = labels[z > z_th]

        # filtered_idx = np.linalg.norm(points, axis=1) >= min_range
        # points = points[filtered_idx]
        # labels = labels[filtered_idx]

        # filter the outliers according to semantic labels
        sem_labels = labels & 0xFFFF
        
        filtered_idx = sem_labels > 1
        points = points[filtered_idx]
        labels = labels[filtered_idx]

        if filter_moving:
            sem_labels = labels & 0xFFFF
            filtered_idx = sem_labels < 100 
            points = points[filtered_idx]
            labels = labels[filtered_idx]

        return points, labels
    
    def write_merged_pc(self, out_path):
        o3d.io.write_point_cloud(out_path, self.map_down_pc) 
        print("save the merged point cloud map to %s\n" % (out_path))    

    def __len__(self) -> int:
        if self.config.ray_loss:
            return self.ray_depth_pool.shape[0]  # ray count
        else:
            return self.sdf_label_pool.shape[0]  # point sample count

    # NOTE: avoid using data loader to load data that are already in gpu
    # deprecated
    def __getitem__(self, index: int):
        # use ray sample (each sample containing all the sample points on the ray)
        if self.config.ray_loss:
            sample_index = torch.range(0, self.ray_sample_count - 1, dtype=int)
            sample_index += index * self.ray_sample_count

            coord = self.coord_pool[sample_index, :]
            # sdf_label = self.sdf_label_pool[sample_index]
            # normal_label = self.normal_label_pool[sample_index]
            # sem_label = self.sem_label_pool[sample_index]
            sample_depth = self.sample_depth_pool[sample_index]
            ray_depth = self.ray_depth_pool[index]

            return coord, sample_depth, ray_depth

        else:  # use point sample
            coord = self.coord_pool[index, :]
            sdf_label = self.sdf_label_pool[index]
            # normal_label = self.normal_label_pool[index]
            # sem_label = self.sem_label_pool[index]
            weight = self.weight_pool[index]

            return coord, sdf_label, weight
    
    def get_batch(self):
        # use ray sample (each sample containing all the sample points on the ray)
        if self.config.ray_loss:
            train_ray_count = self.ray_depth_pool.shape[0]
            ray_index = torch.randint(0, train_ray_count, (self.config.bs,), device=self.config.device)

            ray_index_repeat = (ray_index * self.ray_sample_count).repeat(self.ray_sample_count, 1)
            sample_index = ray_index_repeat + torch.arange(0, self.ray_sample_count,\
                 dtype=int, device=self.config.device).reshape(-1, 1)
            index = sample_index.transpose(0,1).reshape(-1)

            coord = self.coord_pool[index, :]
            weight = self.weight_pool[index]
            sample_depth = self.sample_depth_pool[index]

            if self.normal_label_pool is not None:
                normal_label = self.normal_label_pool[index, :]
            else: 
                normal_label = None

            if self.sem_label_pool is not None:
                sem_label = self.sem_label_pool[ray_index * self.ray_sample_count] # one semantic label for one ray
            else: 
                sem_label = None

            ray_depth = self.ray_depth_pool[ray_index]

            return coord, sample_depth, ray_depth, normal_label, sem_label, weight

        else: # use point sample
            train_sample_count = self.sdf_label_pool.shape[0]
            index = torch.randint(0, train_sample_count, (self.config.bs,), device=self.config.device)
            coord = self.coord_pool[index, :]
            sdf_label = self.sdf_label_pool[index]
            
            if self.normal_label_pool is not None:
                normal_label = self.normal_label_pool[index, :]
            else: 
                normal_label = None
            
            if self.sem_label_pool is not None:
                sem_label = self.sem_label_pool[index]
            else: 
                sem_label = None

            weight = self.weight_pool[index]

            return coord, sdf_label, normal_label, sem_label, weight


