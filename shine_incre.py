import sys
import wandb
import numpy as np
from numpy.linalg import inv, norm
from tqdm import tqdm
import open3d as o3d
import wandb
import torch
import torch.nn as nn
import torch.optim as optim

from utils.config import SHINEConfig
from utils.tools import *
from utils.loss import *
from utils.incre_learning import cal_feature_importance
from utils.mesher import Mesher
from utils.visualizer import MapVisualizer
from model.feature_octree import FeatureOctree
from model.decoder import Decoder
from dataset.lidar_dataset import LiDARDataset

def run_shine_mapping_incremental():

    config = SHINEConfig()
    if len(sys.argv) > 1:
        config.load(sys.argv[1])
    else:
        sys.exit(
            "Please provide the path to the config file.\nTry: python shine_incre.py xxx/xxx_config.yaml"
        )

    run_path = setup_experiment(config)
    dev = config.device

    # initialize the feature octree
    octree = FeatureOctree(config)
    # initialize the mlp decoder
    mlp = Decoder(config)

    # Load the decoder model
    if config.load_model:
        loaded_model = torch.load(config.model_path)
        mlp.load_state_dict(loaded_model["decoder"])
        freeze_model(mlp) # fixed the decoder

    # dataset
    dataset = LiDARDataset(config, octree)

    # mesh reconstructor
    mesher = Mesher(config, octree, mlp)

    # Non-blocking visualizer
    vis = MapVisualizer()

    # learnable parameters
    mlp_param = list(mlp.parameters())
    # learnable sigma for differentiable rendering
    sigma_size = torch.nn.Parameter(torch.ones(1, device=dev)*1.0) 
    # fixed sigma for sdf prediction supervised with BCE loss
    sigma_sigmoid = config.logistic_gaussian_ratio*config.sigma_sigmoid_m*config.scale

    processed_frame = 0
    total_iter = 0

    # for each frame
    for frame_id in tqdm(range(dataset.total_pc_count)):
        if (frame_id < config.begin_frame or frame_id > config.end_frame or \
            frame_id % config.every_frame != 0): 
            continue
        
        vis_mesh = False 

        T0 = get_time()
        # preprocess, sample data and update the octree
        # if continual_learning_reg is on, we only keep the current frame's sample in the data pool,
        # otherwise we accumulate the data pool with the current frame's sample
        dataset.process_frame(frame_id, incremental_on=config.continual_learning_reg)

        if processed_frame == config.freeze_after_frame: # freeze the decoder after certain frame
            print("Freeze the decoder")
            freeze_model(mlp) # fixed the decoder
        
        octree_feat = list(octree.parameters())
        opt = setup_optimizer(config, octree_feat, mlp_param, sigma_size)
        octree.print_detail()

        T1 = get_time()

        for iter in tqdm(range(config.iters)):
            # load batch data (avoid using dataloader because the data are already in gpu, memory vs speed)
            coord, sdf_label, normal_label, weight = dataset.get_batch()
            
            octree.get_indices(coord)
            
            if config.ekional_loss_on:
                coord.requires_grad_()
            
            # interpolate and concat the hierachical grid features
            feature = octree.query_feature(coord) 
            
            # predict the scaled sdf with the feature
            pred = mlp(feature)

            # calculate the loss
            cur_loss = 0.
            weight = torch.abs(weight) # weight's sign indicate the sample is around the surface or in the free space
            sdf_loss = sdf_bce_loss(pred, sdf_label, sigma_sigmoid, weight, config.loss_weight_on, "mean") 
            cur_loss += sdf_loss

            # incremental learning regularization loss 
            reg_loss = 0.
            if config.continual_learning_reg:
                reg_loss = octree.cal_regularization()
                cur_loss += config.lambda_forget * reg_loss

            # optional ekional loss
            eikonal_loss = 0.
            if config.ekional_loss_on:
                surface_mask = weight > 0
                g = gradient(coord, pred)*sigma_sigmoid
                eikonal_loss = ((g[surface_mask].norm(2, dim=-1) - 1.0) ** 2).mean() # MSE with regards to 1  
                cur_loss += config.weight_e * eikonal_loss

            opt.zero_grad(set_to_none=True)
            cur_loss.backward() # this is the slowest part (about 10x the forward time)
            opt.step()

            octree.set_zero() # set the trashbin feature vector back to 0 after the feature update
            total_iter += 1

            if config.wandb_vis_on:
                wandb_log_content = {'iter': total_iter, 'loss/total_loss': cur_loss, 'loss/sdf_loss': sdf_loss, 'loss/reg':reg_loss, 'loss/eikonal_loss': eikonal_loss} 
                wandb.log(wandb_log_content)
        
        # calculate the importance of each octree feature
        if config.continual_learning_reg:
            cal_feature_importance(dataset, octree, mlp, sigma_sigmoid, config.bs, config.cal_importance_weight_down_rate)

        T2 = get_time()

        # reconstruction by marching cubes
        if processed_frame == 0 or (processed_frame+1) % config.mesh_freq_frame == 0:
            vis_mesh = True 
            # print("Begin reconstruction from implicit mapn")               
            mesh_path = run_path + '/mesh/mesh_frame_' + str(frame_id) + ".ply"
            mesher.recon_bbx_mesh(dataset.map_bbx, config.mc_res_m, mesh_path)

        T3 = get_time()

        if vis_mesh: 
            cur_mesh = o3d.io.read_triangle_mesh(mesh_path)
            cur_mesh.compute_vertex_normals()
            vis.update(dataset.cur_frame_pc, dataset.cur_pose_ref, cur_mesh)
        else: # only show frame and current point cloud
            vis.update(dataset.cur_frame_pc, dataset.cur_pose_ref)

        if config.wandb_vis_on:
            wandb_log_content = {'frame': processed_frame, 'timing(s)/preprocess': T1-T0, 'timing(s)/mapping': T2-T1, 'timing(s)/reconstruct': T3-T2} 
            wandb.log(wandb_log_content)

        processed_frame += 1

if __name__ == "__main__":
    run_shine_mapping_incremental()