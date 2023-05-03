import sys
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
from utils.visualizer import MapVisualizer, random_color_table
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
    geo_mlp = Decoder(config, is_geo_encoder=True)
    sem_mlp = Decoder(config, is_geo_encoder=False)

    # Load the decoder model
    if config.load_model:
        loaded_model = torch.load(config.model_path)
        geo_mlp.load_state_dict(loaded_model["geo_decoder"])
        print("Pretrained decoder loaded")
        freeze_model(geo_mlp) # fixed the decoder
        if config.semantic_on:
            sem_mlp.load_state_dict(loaded_model["sem_decoder"])
            freeze_model(sem_mlp) # fixed the decoder
        if 'feature_octree' in loaded_model.keys(): # also load the feature octree  
            octree = loaded_model["feature_octree"]
            octree.print_detail()

    # dataset
    dataset = LiDARDataset(config, octree)

    # mesh reconstructor
    mesher = Mesher(config, octree, geo_mlp, sem_mlp)
    mesher.global_transform = inv(dataset.begin_pose_inv)

    # Non-blocking visualizer
    if config.o3d_vis_on:
        vis = MapVisualizer()

    # learnable parameters
    geo_mlp_param = list(geo_mlp.parameters())
    # learnable sigma for differentiable rendering
    sigma_size = torch.nn.Parameter(torch.ones(1, device=dev)*1.0) 
    # fixed sigma for sdf prediction supervised with BCE loss
    sigma_sigmoid = config.logistic_gaussian_ratio*config.sigma_sigmoid_m*config.scale

    processed_frame = 0
    total_iter = 0
    if config.continual_learning_reg:
        config.loss_reduction = "sum" # other-wise "mean"

    # for each frame
    for frame_id in tqdm(range(dataset.total_pc_count)):
        if (frame_id < config.begin_frame or frame_id > config.end_frame or \
            frame_id % config.every_frame != 0): 
            continue
        
        vis_mesh = False 

        if processed_frame == config.freeze_after_frame: # freeze the decoder after certain frame
            print("Freeze the decoder")
            freeze_model(geo_mlp) # fixed the decoder
            if config.semantic_on:
                freeze_model(sem_mlp) # fixed the decoder

        T0 = get_time()
        # preprocess, sample data and update the octree
        # if continual_learning_reg is on, we only keep the current frame's sample in the data pool,
        # otherwise we accumulate the data pool with the current frame's sample

        local_data_only = False # this one would lead to the forgetting issue

        dataset.process_frame(frame_id, incremental_on=config.continual_learning_reg or local_data_only)
        
        octree_feat = list(octree.parameters())
        opt = setup_optimizer(config, octree_feat, geo_mlp_param, None, sigma_size)
        octree.print_detail()

        T1 = get_time()

        for iter in tqdm(range(config.iters)):
            # load batch data (avoid using dataloader because the data are already in gpu, memory vs speed)

            # we do not use the ray rendering loss here for the incremental mapping
            coord, sdf_label, _, _, _, sem_label, weight = dataset.get_batch() 
            
            if config.normal_loss_on or config.ekional_loss_on:
                coord.requires_grad_(True)

            # interpolate and concat the hierachical grid features
            feature = octree.query_feature(coord)
            
            # predict the scaled sdf with the feature
            sdf_pred = geo_mlp.sdf(feature)
            if config.semantic_on:
                sem_pred = sem_mlp.sem_label_prob(feature)

            # calculate the loss
            surface_mask = weight > 0
            cur_loss = 0.
            weight = torch.abs(weight) # weight's sign indicate the sample is around the surface or in the free space
            sdf_loss = sdf_bce_loss(sdf_pred, sdf_label, sigma_sigmoid, weight, config.loss_weight_on, config.loss_reduction) 
            cur_loss += sdf_loss

            # incremental learning regularization loss 
            reg_loss = 0.
            if config.continual_learning_reg:
                reg_loss = octree.cal_regularization()
                cur_loss += config.lambda_forget * reg_loss

            # optional ekional loss
            eikonal_loss = 0.
            if config.ekional_loss_on:
                g = get_gradient(coord, sdf_pred)*sigma_sigmoid
                eikonal_loss = ((g[surface_mask].norm(2, dim=-1) - 1.0) ** 2).mean() # MSE with regards to 1  
                cur_loss += config.weight_e * eikonal_loss
            
            # semantic classification loss
            sem_loss = 0.
            if config.semantic_on:
                loss_nll = nn.NLLLoss(reduction='mean')
                sem_loss = loss_nll(sem_pred[::config.sem_label_decimation,:], sem_label[::config.sem_label_decimation])
                cur_loss += config.weight_s * sem_loss

            opt.zero_grad(set_to_none=True)
            cur_loss.backward() # this is the slowest part (about 10x the forward time)
            opt.step()

            total_iter += 1

            if config.wandb_vis_on:
                wandb_log_content = {'iter': total_iter, 'loss/total_loss': cur_loss, 'loss/sdf_loss': sdf_loss, \
                    'loss/reg_loss':reg_loss, 'loss/eikonal_loss': eikonal_loss, 'loss/sem_loss': sem_loss} 
                wandb.log(wandb_log_content)
        
        # calculate the importance of each octree feature
        if config.continual_learning_reg:
            opt.zero_grad(set_to_none=True)
            cal_feature_importance(dataset, octree, geo_mlp, sigma_sigmoid, config.bs, \
                config.cal_importance_weight_down_rate, config.loss_reduction)


        T2 = get_time()
        
        # reconstruction by marching cubes
        if processed_frame == 0 or (processed_frame+1) % config.mesh_freq_frame == 0:
            print("Begin mesh reconstruction from the implicit map")       
            vis_mesh = True 
            # print("Begin reconstruction from implicit mapn")               
            mesh_path = run_path + '/mesh/mesh_frame_' + str(frame_id+1) + ".ply"
            map_path = run_path + '/map/sdf_map_frame_' + str(frame_id+1) + ".ply"
            if config.mc_with_octree: # default
                cur_mesh = mesher.recon_octree_mesh(config.mc_query_level, config.mc_res_m, mesh_path, map_path, config.save_map, config.semantic_on)
            else:
                cur_mesh = mesher.recon_bbx_mesh(dataset.map_bbx, config.mc_res_m, mesh_path, map_path, config.save_map, config.semantic_on)

        T3 = get_time()

        if config.o3d_vis_on:
            if vis_mesh: 
                cur_mesh.transform(dataset.begin_pose_inv) # back to the globally shifted frame for vis
                vis.update(dataset.cur_frame_pc, dataset.cur_pose_ref, cur_mesh)
            else: # only show frame and current point cloud
                vis.update(dataset.cur_frame_pc, dataset.cur_pose_ref)

            # visualize the octree (it is a bit slow and memory intensive for the visualization)
            # if vis_mesh: 
            #     cur_mesh.transform(dataset.begin_pose_inv)
            #     vis_list = [] # create a list of bbx for the octree nodes
            #     for l in range(config.tree_level_feat):
            #         nodes_coord = octree.get_octree_nodes(config.tree_level_world-l)/config.scale
            #         box_size = np.ones(3) * config.leaf_vox_size * (2**l)
            #         for node_coord in nodes_coord:
            #             node_box = o3d.geometry.AxisAlignedBoundingBox(node_coord-0.5*box_size, node_coord+0.5*box_size)
            #             node_box.color = random_color_table[l]
            #             vis_list.append(node_box)
            #     vis_list.append(cur_mesh)
            #     o3d.visualization.draw_geometries(vis_list)

        if config.wandb_vis_on:
            wandb_log_content = {'frame': processed_frame, 'timing(s)/preprocess': T1-T0, 'timing(s)/mapping': T2-T1, 'timing(s)/reconstruct': T3-T2} 
            wandb.log(wandb_log_content)

        processed_frame += 1
    
    if config.o3d_vis_on:
        vis.stop()

if __name__ == "__main__":
    run_shine_mapping_incremental()