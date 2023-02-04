import numpy as np
from tqdm import tqdm
import skimage.measure
import torch
import math
import open3d as o3d
import copy
import kaolin as kal
from utils.config import SHINEConfig
from utils.semantic_kitti_utils import *
from utils.tools import get_gradient
from model.feature_octree import FeatureOctree
from model.decoder import Decoder


class Tracker():

    def __init__(self, config: SHINEConfig, octree: FeatureOctree, geo_decoder: Decoder, sem_decoder: Decoder):
    
        self.config = config
    
        self.octree = octree
        self.geo_decoder = geo_decoder
        self.sem_decoder = sem_decoder
        self.device = config.device
        self.dtype = config.dtype
        self.world_scale = config.scale
        self.sdf_scale = config.logistic_gaussian_ratio*config.sigma_sigmoid_m 

    def tracking(self, source_scan, init_pose, iters: int = 10):        
        tran = init_pose
        cur_tran = init_pose
        scan = copy.deepcopy(source_scan)
        for i in tqdm(range(iters)):
            
            scan = scan.transform(tran)
            source_points = np.asarray(scan.points)
            source_points_torch = torch.tensor(source_points, dtype=self.dtype, device=self.device) * self.world_scale
            sdf_pred, sdf_grad_unit, _, mask = self.query_points(source_points_torch, self.config.infer_bs)
            source_points = source_points[mask > 0]
            sdf_pred = sdf_pred[mask > 0] * self.sdf_scale
            sdf_grad_unit = sdf_grad_unit[mask > 0]
            df_pred = np.expand_dims(sdf_pred, axis=1)
            # print(df_pred)
            
            target_points = source_points - df_pred * sdf_grad_unit

            scan.points = o3d.utility.Vector3dVector(source_points)
            target_pcd = o3d.geometry.PointCloud()
            target_pcd.points = o3d.utility.Vector3dVector(target_points)

            tran = self.register_step(scan, target_pcd, 2.0) # unit: m
            cur_tran = tran @ cur_tran 

            # self.draw_registration_result(scan, target_pcd) # before reg # scan in yellow, target_pcd in cyan
            # self.draw_registration_result(scan, target_pcd, tran) # after reg

            # print(cur_tran)
        # self.draw_registration_result(scan, target_pcd, tran)

        return cur_tran

    def query_points(self, coord, bs, query_sdf = True, query_gradient = True, query_sem = False, query_mask = True):
        """ query the sdf value, semantic label and marching cubes mask for points
        Args:
            coord: Nx3 torch tensor, the coordinates of all N (axbxc) query points in the scaled
                kaolin coordinate system [-1,1]
            bs: batch size for the inference
        Returns:
            sdf_pred: Ndim numpy array, signed distance value (scaled) at each query point
            sem_pred: Ndim numpy array, semantic label prediction at each query point
            mc_mask:  Ndim bool numpy array, marching cubes mask at each query point
        """
        
        sample_count = coord.shape[0]
        iter_n = math.ceil(sample_count/bs)
        # check_level = min(self.octree.featured_level_num, self.config.mc_vis_level)-1
        check_level = 2
        if query_sdf:
            sdf_pred = np.zeros(sample_count)
        else: 
            sdf_pred = None
        if query_sem:
            sem_pred = np.zeros(sample_count)
        else:
            sem_pred = None
        if query_mask:
            mc_mask = np.zeros(sample_count)
        else:
            mc_mask = None
        if query_gradient:
            sdf_grad_unit = np.zeros((sample_count, 3))
        else:
            query_gradient = None
        
        for n in tqdm(range(iter_n)):
            head = n*bs
            tail = min((n+1)*bs, sample_count)
            batch_coord = coord[head:tail, :]
            if query_gradient:
                batch_coord.requires_grad_(True)

            batch_feature = self.octree.query_feature(batch_coord)
            if query_sdf:
                batch_sdf = -self.geo_decoder.sdf(batch_feature)
                if query_gradient:
                    batch_grad = get_gradient(batch_coord, batch_sdf)
                    batch_grad_norm = batch_grad.norm(2, dim=-1).view(-1,1)
                    batch_grad_direction = batch_grad / batch_grad_norm
                    # print(batch_grad_direction)
                    sdf_grad_unit[head:tail, :] = batch_grad_direction.detach().cpu().numpy()
                sdf_pred[head:tail] = batch_sdf.detach().cpu().numpy()  
            if query_sem:
                batch_sem = self.sem_decoder.sem_label(batch_feature)
                sem_pred[head:tail] = batch_sem.detach().cpu().numpy()
            if query_mask:
                # get the marching cubes mask
                # hierarchical_indices: bottom-up
                check_level_indices = self.octree.hierarchical_indices[check_level] 
                # if index is -1 for the level, then means the point is not valid under this level
                mask_mc = check_level_indices >= 0
                # print(mask_mc.shape)
                # all should be true (all the corner should be valid)
                mask_mc = torch.all(mask_mc, dim=1)
                mc_mask[head:tail] = mask_mc.detach().cpu().numpy()
                # but for scimage's marching cubes, the top right corner's mask should also be true to conduct marching cubes

        return sdf_pred, sdf_grad_unit, sem_pred, mc_mask    
    
    def register_step(self, source, target, threshold, robust_on = False, sigma = 0.1):
        if robust_on:
            loss = o3d.pipelines.registration.TukeyLoss(k=sigma)
        else:
            loss = None
        p2p = o3d.pipelines.registration.TransformationEstimationPointToPoint(loss)
        criteria = o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration = 1)
        reg_p2p = o3d.pipelines.registration.registration_icp(source, target, threshold,
                                                              np.eye(4), p2p, criteria)
        
        trans = reg_p2p.transformation
        return trans 

    def draw_registration_result(self, source, target, transformation = np.eye(4)):
        source_temp = copy.deepcopy(source)
        target_temp = copy.deepcopy(target)
        source_temp.paint_uniform_color([1, 0.706, 0])
        target_temp.paint_uniform_color([0, 0.651, 0.929])
        source_temp.transform(transformation)
        o3d.visualization.draw_geometries([source_temp, target_temp])                                  
