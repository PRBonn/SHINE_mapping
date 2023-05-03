import numpy as np
from numpy.linalg import inv, norm
from scipy.fftpack import shift
import kaolin as kal
import torch

from utils.config import SHINEConfig

class dataSampler():

    def __init__(self, config: SHINEConfig):

        self.config = config
        self.dev = config.device


    # input and output are all torch tensors
    def sample(self, points_torch, 
               sensor_origin_torch,
               normal_torch,
               sem_label_torch):

        dev = self.dev

        world_scale = self.config.scale
        surface_sample_range_scaled = self.config.surface_sample_range_m * self.config.scale
        surface_sample_n = self.config.surface_sample_n
        clearance_sample_n = self.config.clearance_sample_n # new part

        freespace_sample_n = self.config.free_sample_n
        all_sample_n = surface_sample_n+clearance_sample_n+freespace_sample_n
        free_min_ratio = self.config.free_sample_begin_ratio
        free_sample_end_dist_m_scaled = self.config.free_sample_end_dist_m * self.config.scale
        clearance_dist_scaled = self.config.clearance_dist_m * self.config.scale
        
        sigma_base = self.config.sigma_sigmoid_m * self.config.scale
        # sigma_scale_constant = self.config.sigma_scale_constant

        # get sample points
        shift_points = points_torch - sensor_origin_torch
        point_num = shift_points.shape[0]
        distances = torch.linalg.norm(shift_points, dim=1, keepdim=True) # ray distances (scaled)
        
        # Part 1. close-to-surface uniform sampling 
        # uniform sample in the close-to-surface range (+- range)
        surface_sample_displacement = (torch.rand(point_num*surface_sample_n, 1, device=dev)-0.5)*2*surface_sample_range_scaled 
        
        repeated_dist = distances.repeat(surface_sample_n,1)
        surface_sample_dist_ratio = surface_sample_displacement/repeated_dist + 1.0 # 1.0 means on the surface
        if sem_label_torch is not None:
            surface_sem_label_tensor = sem_label_torch.repeat(1, surface_sample_n).transpose(0,1)
        
        # Part 2. near surface uniform sampling (for clearance) [from the close surface lower bound closer to the sensor for a clearance distance]
        clearance_sample_displacement = -torch.rand(point_num*clearance_sample_n, 1, device=dev)*clearance_dist_scaled - surface_sample_range_scaled

        repeated_dist = distances.repeat(clearance_sample_n,1)
        clearance_sample_dist_ratio = clearance_sample_displacement/repeated_dist + 1.0 # 1.0 means on the surface
        if sem_label_torch is not None:
            clearance_sem_label_tensor = torch.zeros_like(repeated_dist)

        # Part 3. free space uniform sampling
        repeated_dist = distances.repeat(freespace_sample_n,1)
        free_max_ratio = free_sample_end_dist_m_scaled / repeated_dist + 1.0
        free_diff_ratio = free_max_ratio - free_min_ratio

        free_sample_dist_ratio = torch.rand(point_num*freespace_sample_n, 1, device=dev)*free_diff_ratio + free_min_ratio
        
        free_sample_displacement = (free_sample_dist_ratio - 1.0) * repeated_dist
        if sem_label_torch is not None:
            free_sem_label_tensor = torch.zeros_like(repeated_dist)
        
        # all together
        all_sample_displacement = torch.cat((surface_sample_displacement, clearance_sample_displacement, free_sample_displacement),0)
        all_sample_dist_ratio = torch.cat((surface_sample_dist_ratio, clearance_sample_dist_ratio, free_sample_dist_ratio),0)
        
        repeated_points = shift_points.repeat(all_sample_n,1)
        repeated_dist = distances.repeat(all_sample_n,1)
        all_sample_points = repeated_points*all_sample_dist_ratio + sensor_origin_torch

        # depth tensor of all the samples
        depths_tensor = repeated_dist * all_sample_dist_ratio
        depths_tensor /= world_scale # unit: m

        # linear error model: sigma(d) = sigma_base + d * sigma_scale_constant
        # ray_sigma = sigma_base + distances * sigma_scale_constant  
        # different sigma value for different ray with different distance (deprecated)
        # sigma_tensor = ray_sigma.repeat(all_sample_n,1).squeeze(1)

        # get the weight vector as the inverse of sigma
        weight_tensor = torch.ones_like(depths_tensor)

        # behind surface weight drop-off because we have less uncertainty behind the surface
        if self.config.behind_dropoff_on:
            dropoff_min = self.config.dropoff_min_sigma
            dropoff_max = self.config.dropoff_max_sigma
            dropoff_diff = dropoff_max - dropoff_min
            behind_displacement = (repeated_dist*(all_sample_dist_ratio-1.0)/sigma_base).squeeze(1)
            dropoff_weight = (dropoff_max - behind_displacement) / dropoff_diff
            dropoff_weight = torch.clamp(dropoff_weight, min = 0.0, max = 1.0)
            weight_tensor *= dropoff_weight
        
        # give a flag indicating the type of the sample [negative: freespace, positive: surface]
        weight_tensor[point_num*surface_sample_n:] *= -1.0 
        
        # ray-wise depth
        distances /= world_scale # unit: m
        distances = distances.squeeze(1)

        # assign sdf labels to the samples
        # projective distance as the label: behind +, in-front - 
        sdf_label_tensor = all_sample_displacement.squeeze(1)  # scaled [-1, 1] # as distance (before sigmoid)

        # assign the normal label to the samples
        normal_label_tensor = None
        if normal_torch is not None:
            normal_label_tensor = normal_torch.repeat(all_sample_n,1)
        
        # assign the semantic label to the samples (including free space as the 0 label)
        sem_label_tensor = None
        if sem_label_torch is not None:
            sem_label_tensor = torch.cat((surface_sem_label_tensor, clearance_sem_label_tensor, free_sem_label_tensor),0).int()

        # Convert from the all ray surface + all ray free order to the 
        # ray-wise (surface + free) order
        all_sample_points = all_sample_points.reshape(all_sample_n, -1, 3).transpose(0, 1).reshape(-1, 3)
        sdf_label_tensor = sdf_label_tensor.reshape(all_sample_n, -1).transpose(0, 1).reshape(-1) 
        
        weight_tensor = weight_tensor.reshape(all_sample_n, -1).transpose(0, 1).reshape(-1)
        depths_tensor = depths_tensor.reshape(all_sample_n, -1).transpose(0, 1).reshape(-1)

        if normal_torch is not None:
            normal_label_tensor = normal_label_tensor.reshape(all_sample_n, -1, 3).transpose(0, 1).reshape(-1, 3)
        if sem_label_torch is not None:
            sem_label_tensor = sem_label_tensor.reshape(all_sample_n, -1).transpose(0, 1).reshape(-1)

        # ray distance (distances) is not repeated

        return all_sample_points, sdf_label_tensor, normal_label_tensor, sem_label_tensor, \
            weight_tensor, depths_tensor, distances
    

    # space carving sampling (deprecated, to polish)
    def sapce_carving_sample(self, 
                             points_torch, 
                             sensor_origin_torch,
                             space_carving_level,
                             stop_depth_thre,
                             inter_dist_thre):
        
        shift_points = points_torch - sensor_origin_torch
        # distances = torch.linalg.norm(shift_points, dim=1, keepdim=True)
        spc = kal.ops.conversions.unbatched_pointcloud_to_spc(shift_points, space_carving_level)

        shift_points_directions = (shift_points/(shift_points**2).sum(1).sqrt().reshape(-1,1))
        virtual_origin = -shift_points_directions*3
            
        octree, point_hierarchy, pyramid, prefix = spc.octrees, spc.point_hierarchies, spc.pyramids[0], spc.exsum
        nugs_ridx, nugs_pidx, depth = kal.render.spc.unbatched_raytrace(octree, point_hierarchy, pyramid, prefix, \
                                                                            virtual_origin, shift_points_directions, space_carving_level, with_exit=True)

        stop_depth =  (shift_points**2).sum(1).sqrt() - stop_depth_thre + 3.0
        mask = (depth[:,0]>3.0) & (depth[:,1]<stop_depth[nugs_ridx.long()]) & ((depth[:,1] - depth[:,0])> inter_dist_thre)
   
        steps = torch.rand(mask.sum().item(),1).cuda() # randomly sample one point on each intersected segment 
        origins = virtual_origin[nugs_ridx[mask].long()]
        directions = shift_points_directions[nugs_ridx[mask].long()]
        depth_range = depth[mask,1] - depth[mask,0]

        space_carving_samples = origins + directions*((depth[mask,0] + steps.reshape(1,-1)*depth_range).reshape(-1,1))

        space_carving_labels = torch.zeros(space_carving_samples.shape[0], device=self.dev) # all as 0 (free)

        return space_carving_samples, space_carving_labels