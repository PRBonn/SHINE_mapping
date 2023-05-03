import yaml
import os
import torch
from typing import List

class SHINEConfig:
    def __init__(self):

        # Default values

        # settings
        self.name: str = "dummy"  # experiment name

        self.output_root: str = ""  # output root folder
        self.pc_path: str = ""  # input point cloud folder
        self.pose_path: str = ""  # input pose file
        self.calib_path: str = ""  # input calib file (to sensor frame)

        self.label_path: str = "" # input point-wise label path, for semantic shine mapping

        self.load_model: bool = False  # load the pre-trained model or not
        self.model_path: str = "/"  # pre-trained model path

        self.first_frame_ref: bool = True  # if false, we directly use the world
        # frame as the reference frame
        self.begin_frame: int = 0  # begin from this frame
        self.end_frame: int = 0  # end at this frame
        self.every_frame: int = 1  # process every x frame

        self.num_workers: int = 12 # number of worker for the dataloader
        self.device: str = "cuda"  # use "cuda" or "cpu"
        self.gpu_id: str = "0"  # used GPU id
        self.dtype = torch.float32 # default torch tensor data type
        self.pc_count_gpu_limit: int = 500 # maximum used frame number to be stored in the gpu

        # just a ramdom number for the global shift of the input on z axis (used to avoid octree boundary marching cubes issues)
        self.global_shift_default: float = 0.17241 

        # baseline
        # self.run_baseline = False
        # # select from vdb_fusion, voxblox_simple, voxblox_merged, voxblox_fast
        # self.baseline_method = "vdb_fusion"
        # self.voxel_size_m = 0.2
        # self.sdf_trunc_m = 3 * self.voxel_size_m

        # process
        self.min_range: float = 2.75 # filter too-close points (and 0 artifacts)
        self.pc_radius: float = 20.0  # keep only the point cloud inside the
        # block with such radius (unit: m)
        self.min_z: float = -3.0  # filter for z coordinates (unit: m)
        self.max_z: float = 30.0

        self.rand_downsample: bool = (
            True  # apply random or voxel downsampling to input original point clcoud
        )
        self.vox_down_m: float = (
            0.03  # the voxel size if using voxel downsampling (unit: m)
        )
        self.rand_down_r: float = (
            1.0  # the decimation ratio if using random downsampling (0-1)
        )

        self.filter_noise: bool = False  # use SOR to remove the noise or not
        self.sor_nn: int = 25  # SOR neighborhood size
        self.sor_std: float = 2.5  # SOR std threshold

        self.estimate_normal: bool = False  # estimate surface normal or not
        self.normal_radius_m: float = 0.2  # supporting radius for estimating the normal
        self.normal_max_nn: int = (
            20  # supporting neighbor count for estimating the normal
        )

        # semantic related
        self.semantic_on: bool = False # semantic shine mapping on [semantic]
        self.sem_class_count: int = 20 # semantic class count: 20 for semantic kitti
        self.sem_label_decimation: int = 1 # use only 1/${sem_label_decimation} of the available semantic labels for training (fitting)
        self.filter_moving_object: bool = False

        # frame-wise downsampling voxel size for the merged map point cloud (unit: m)
        self.map_vox_down_m: float = 0.05 # 0.2 

        # octree
        self.tree_level_world: int = (
            10  # the total octree level, allocated for the whole space
        )
        self.tree_level_feat: int = 4  # the octree levels with optimizable feature grid
        # start from the leaf level
        self.leaf_vox_size: float = 0.5  # voxel size of the octree leaf nodes (unit: m)
        self.feature_dim: int = 8  # length of the feature for each grid feature
        self.feature_std: float = 0.05  # grid feature initialization standard deviation
        self.poly_int_on: bool = (
            True  # use polynomial interpolation or linear interpolation
        )
        self.octree_from_surface_samples: bool = True  # Use all the surface samples or just the exact measurements to build the octree. If True may lead to larger memory, but is more robust while the reconstruction.

        # sampler
        # spilt into 3 parts for sampling
        self.surface_sample_range_m: float = 0.5 # 
        self.surface_sample_n: int = 5
        self.free_sample_begin_ratio: float = 0.3
        # self.free_sample_end_ratio: float = 1.0 # deprecated
        self.free_sample_end_dist_m: float = 0.5 # maximum distance after the surface (unit: m)
        self.free_sample_n: int = 2
        self.clearance_dist_m: float = 0.3
        self.clearance_sample_n: int = 0

        # space carving sampling related (deprecated)
        # self.carving_on = False
        # self.tree_level_carving = self.tree_level_world
        # self.carving_stop_depth_m = 0.5
        # self.carving_inte_thre_m = 0.1

        # incremental mapping
        self.continual_learning_reg: bool = True
        # regularization based
        self.lambda_forget: float = 1e5
        self.cal_importance_weight_down_rate: int = 10 # set it larger to save the consuming time
        
        # replay based
        self.window_replay_on: bool = True
        self.window_radius: float = 50.0 # unit: m

        # label
        self.occu_update_on: bool = False

        # decoder
        self.geo_mlp_level: int = 2
        self.geo_mlp_hidden_dim: int = 32
        self.geo_mlp_bias_on: bool = True

        self.sem_mlp_level: int = 2
        self.sem_mlp_hidden_dim: int = 32
        self.sem_mlp_bias_on: bool = True
        
        self.freeze_after_frame: int = 20  # For incremental mode only, if the decoder model is not loaded , it would be trained and freezed after such frame number

        # loss
        self.ray_loss: bool = False  # one loss on a whole ray (including depth estimation loss or the differentiable rendering loss)
        # the main loss type, select from the sample sdf loss ('sdf_bce', 'sdf_l1', 'sdf_l2') and the ray rendering loss ('dr', 'dr_neus')
        self.main_loss_type: str = 'sdf_bce'

        self.loss_reduction: str = 'mean' # select from 'mean' and 'sum' (for incremental mapping)
        
        self.sigma_sigmoid_m: float = 0.1
        self.sigma_scale_constant: float = 0.0 # scale factor adding to the constant sigma value (linear with the distance) [deprecated]
        self.logistic_gaussian_ratio: float = 0.55

        self.proj_correction_on: bool = False # conduct projective distance correction based on the sdf gradient or not
        
        self.predict_sdf: bool = False
        self.neus_loss_on: bool = False  # use the unbiased and occlusion-aware weights for differentiable rendering as introduced in NEUS
        self.loss_weight_on: bool = False  # if True, the weight would be given to the loss, if False, the weight would be used to change the sigmoid's shape
        self.behind_dropoff_on: bool = False  # behind surface drop off weight
        self.dropoff_min_sigma: float = 1.0
        self.dropoff_max_sigma: float = 5.0
        self.normal_loss_on: bool = False
        self.weight_n: float = 0.01
        self.ekional_loss_on: bool = False
        self.weight_e: float = 0.1

        # TODO: add to config file
        self.consistency_loss_on: bool = False
        self.weight_c: float = 1.0
        self.consistency_count: int = 1000
        self.consistency_range: float = 0.1 # the neighborhood points would be randomly select within the radius of xxx m
        
        self.history_weight: float = 1.0

        self.weight_s: float = 1.0  # weight for semantic classification loss

        # for dynamic reconstruction (TODO)
        self.time_conditioned: bool = False

        # optimizer
        self.iters: int = 200
        self.opt_adam: bool = True  # use adam or sgd
        self.bs: int = 4096
        self.lr: float = 1e-3
        self.weight_decay: float = 0
        self.adam_eps: float = 1e-15
        self.lr_level_reduce_ratio: float = 1.0
        self.lr_iters_reduce_ratio: float = 0.1
        self.lr_decay_step: List = [10000, 50000, 100000]
        self.dropout: float = 0

        # eval
        self.wandb_vis_on: bool = False
        self.o3d_vis_on: bool = True # visualize the mesh in-the-fly using o3d visualzier or not [press space to pasue/resume]
        self.eval_on: bool = False
        self.eval_outlier_thre = 0.5  # unit:m
        self.eval_freq_iters: int = 100
        self.vis_freq_iters: int = 100
        self.save_freq_iters: int = 100
        self.mesh_freq_frame: int = 1  # do the reconstruction per x frames
        
        # marching cubes related
        self.mc_res_m: float = 0.1
        self.pad_voxel: int = 0
        self.mc_with_octree: bool = True # conducting marching cubes reconstruction within a certain level of the octree or within the axis-aligned bounding box of the whole map
        self.mc_query_level: int = 8
        self.mc_vis_level: int = 1 # masked the marching cubes for level higher than this
        self.mc_mask_on: bool = True # use mask for marching cubes to avoid the artifacts

        self.min_cluster_vertices: int = 300 # if a connected's vertices number is smaller than this value, it would get filtered
        
        self.infer_bs: int = 4096
        self.occ_binary_mc: bool = False
        self.grid_loss_vis_on: bool = False
        self.mesh_vis_on: bool = True
        self.save_map: bool = False # save the sdf map or not, the sdf would be saved in the intensity channel

        # initialization
        self.scale: float = 1.0
        self.world_size: float = 1.0

    def load(self, config_file):
        config_args = yaml.safe_load(open(os.path.abspath(config_file)))

        # common
        self.name = config_args["setting"]["name"] 
        
        self.output_root = config_args["setting"]["output_root"]  
        self.pc_path = config_args["setting"]["pc_path"] 
        self.pose_path = config_args["setting"]["pose_path"]
        self.calib_path = config_args["setting"]["calib_path"]

        # optional, when semantic shine mapping is on [semantic]
        if self.semantic_on:
            self.label_path =  config_args["setting"]["label_path"] 

        self.load_model = config_args["setting"]["load_model"]
        self.model_path = config_args["setting"]["model_path"]
        
        self.first_frame_ref = config_args["setting"]["first_frame_ref"]
        self.begin_frame = config_args["setting"]["begin_frame"]
        self.end_frame = config_args["setting"]["end_frame"]
        self.every_frame = config_args["setting"]["every_frame"]

        self.device = config_args["setting"]["device"]
        self.gpu_id = config_args["setting"]["gpu_id"]

        # process
        self.min_range = config_args["process"]["min_range_m"]
        self.pc_radius = config_args["process"]["pc_radius_m"]
        self.rand_downsample = config_args["process"]["rand_downsample"]
        self.vox_down_m = config_args["process"]["vox_down_m"]
        self.rand_down_r = config_args["process"]["rand_down_r"]
        # self.estimate_normal = config_args["process"]["estimate_normal"]
        # self.filter_noise = config_args["process"]["filter_noise"]
        # self.semantic_on = config_args["process"]["semantic_on"] 

        # sampler
        self.surface_sample_range_m = config_args["sampler"]["surface_sample_range_m"]
        self.surface_sample_n = config_args["sampler"]["surface_sample_n"]
        self.free_sample_begin_ratio = config_args["sampler"]["free_sample_begin_ratio"]
        self.free_sample_end_dist_m = config_args["sampler"]["free_sample_end_dist_m"]
        self.free_sample_n = config_args["sampler"]["free_sample_n"]
        # optional split
        # self.clearance_dist_m = config_args["sampler"]["clearance_dist_m"]
        # self.clearance_sample_n = config_args["sampler"]["clearance_sample_n"]

        # label
        # self.occu_update_on = config_args["label"]["occu_update_on"]
        # use bayersian update of the occupancy prob. as the new label

        # octree
        self.tree_level_world = config_args["octree"]["tree_level_world"]
        # the number of the total octree level (defining the world scale)
        self.tree_level_feat = config_args["octree"][
            "tree_level_feat"
        ]  # the number of the octree level used for storing feature grid
        self.leaf_vox_size = config_args["octree"][
            "leaf_vox_size"
        ]  # the size of the grid on octree's leaf level (unit: m)
        self.feature_dim = config_args["octree"][
            "feature_dim"
        ]  # feature vector's dimension
        # self.feature_std = config_args["octree"][
        #     "feature_std"
        # ]  # feature vector's initialization sigma (a zero mean, sigma standard deviation gaussian distribution)
        self.poly_int_on = config_args["octree"][
            "poly_int_on"
        ]  # use polynomial or linear interpolation of feature grids
        self.octree_from_surface_samples = config_args["octree"][
            "octree_from_surface_samples"
        ]  # build the octree from the surface samples or only the measurement points

        # decoder
        self.geo_mlp_level = config_args["decoder"][
            "mlp_level"
        ]  # number of the level of the mlp decoder
        self.geo_mlp_hidden_dim = config_args["decoder"][
            "mlp_hidden_dim"
        ]  # dimension of the mlp's hidden layer
        # freeze the decoder after runing for x frames (used for incremental mapping to avoid forgeting)
        self.freeze_after_frame = config_args["decoder"]["freeze_after_frame"]

        # do the prediction conditioned on time (frame ID)
        # self.time_conditioned = config_args["decoder"][ 
        #     "time_conditioned"
        # ]

        # loss
        self.ray_loss = config_args["loss"]["ray_loss"]
        self.main_loss_type = config_args["loss"]["main_loss_type"]
        self.sigma_sigmoid_m = config_args["loss"]["sigma_sigmoid_m"]

        self.loss_weight_on = config_args["loss"]["loss_weight_on"]
        
        self.behind_dropoff_on = config_args["loss"][
            "behind_dropoff_on"
        ]  # apply "behind the surface" loss weight drop-off or not

        # self.normal_loss_on = config_args["loss"][
        #     "normal_loss_on"
        # ]  # use normal consistency loss [deprecated]
        # self.weight_n = float(config_args["loss"]["weight_n"])
        
        self.ekional_loss_on = config_args["loss"][
            "ekional_loss_on"
        ]  # use ekional loss (gradient = 1 loss)
        self.weight_e = float(config_args["loss"]["weight_e"])

        
        # continual learning (incremental)
        # using the regularization based continuous learning or the rehersal based continuous learning
        self.continual_learning_reg = config_args["continual"]["continual_learning_reg"]
        # the forgeting lambda for regularization based continual learning
        self.lambda_forget = float(
            config_args["continual"]["lambda_forget"]
        )

        # rehersal (replay) based method
        self.window_replay_on = config_args["continual"]["window_replay_on"]
        self.window_radius = config_args["continual"]["window_radius_m"]

        # self.history_sample_ratio = float(
        #     config_args["continuous"]["history_sample_ratio"]
        # )  # sample the history samples by a scale of the number of current samples
        # self.history_sample_res = config_args["continuous"][
        #     "history_sample_res"
        # ]  # the resolution of the kept history samples (unit: m)
        

        # optimizer
        self.iters = config_args["optimizer"][
            "iters"
        ]  # maximum iters (in our implementation, iters means iteration actually)
        self.bs = config_args["optimizer"]["batch_size"]
        # self.adam_eps = float(config_args["optimizer"]["adam_eps"])
        self.lr = float(config_args["optimizer"]["learning_rate"])
        # self.lr_level_reduce_ratio = config_args["optimizer"][
        #     "lr_level_reduce_ratio"
        # ]  # decay the learning rate for higher level of feature grids by such ratio
        # self.lr_iters_reduce_ratio = config_args["optimizer"][
        #     "lr_iters_reduce_ratio"
        # ]  # decay the learning rate after certain iterss by such ratio
        self.weight_decay = float(
            config_args["optimizer"]["weight_decay"]
        )  # coefficient for L2 regularization
        

        # vis and eval
        self.wandb_vis_on = config_args["eval"][
            "wandb_vis_on"
        ]  # use weight and bias to monitor the experiment or not
        self.o3d_vis_on = config_args["eval"][
            "o3d_vis_on"
        ] # turn on the open3d visualizer to visualize the mapping progress or not
        self.vis_freq_iters = config_args["eval"][
            "vis_freq_iters"
        ]  # frequency for mesh reconstruction for batch mode (per x iters)
        self.save_freq_iters = config_args["eval"][
            "save_freq_iters"
        ]  # frequency for model saving for batch mode (per x iters)
        self.mesh_freq_frame = config_args["eval"][
            "mesh_freq_frame"
        ]  # frequency for mesh reconstruction for incremental mode (per x frame)
        self.mc_with_octree = config_args["eval"][
            "mc_with_octree"
        ]  # using octree to narrow down the region that needs the sdf query so as to boost the efficieny
        # if false, we query all the positions within the map bounding box
        self.mc_res_m = config_args["eval"][
            "mc_res_m"
        ]  # marching cubes grid sampling interval (unit: m)
        self.mc_vis_level = config_args["eval"][
            "mc_vis_level"
        ]
        # self.mc_mask_on = config_args["eval"]["mc_mask_on"] # using masked marching cubes according to the octree or not, default true
        
        self.save_map = config_args["eval"][
            "save_map"
        ] 
        # tree level starting for reconstruction and visualization, the larger of this value, 
        # the larger holes would be filled (better completion), but at the same time more artifacts 
        # would appear at the boundary of the map
        # it's a trading-off of the compeltion and the artifacts

        # self.grid_loss_vis_on = config_args["eval"][
        #     "grid_loss_vis_on"
        # ]  # visualize the loss at each grid position or not [deprecated]
        # self.mesh_vis_on = config_args["eval"][
        #     "mesh_vis_on"
        # ]  # visualize the reconstructed mesh or not, if not, the mesh will still be exported and you can check it offline

        self.calculate_world_scale()
        self.infer_bs = self.bs * 16
        self.mc_query_level = self.tree_level_world - self.tree_level_feat + 1

        if self.window_radius <= 0:
            self.window_radius = self.pc_radius * 2.0
    
    # calculate the scale for compressing the world into a [-1,1] kaolin cube
    def calculate_world_scale(self):
        self.world_size = self.leaf_vox_size*(2**(self.tree_level_world-1)) 
        self.scale = 1.0 / self.world_size
