from typing import List
import sys
import os
import multiprocessing
import getpass
import time
from datetime import datetime
from torch import optim
from torch.optim.optimizer import Optimizer
from torch.autograd import grad
import torch
import torch.nn as nn
import numpy as np
import wandb
import open3d as o3d

from utils.config import SHINEConfig


# setup this run
def setup_experiment(config: SHINEConfig): 

    os.environ["NUMEXPR_MAX_THREADS"] = str(multiprocessing.cpu_count())
    os.environ["CUDA_VISIBLE_DEVICES"] = str(config.gpu_id)
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")  # begining timestamp
    run_name = config.name + "_" + ts  # modified to a name that is easier to index
        
    run_path = os.path.join(config.output_root, run_name)
    access = 0o755
    os.makedirs(run_path, access, exist_ok=True)
    assert os.access(run_path, os.W_OK)
    print(f"Start {run_path}")

    mesh_path = os.path.join(run_path, "mesh")
    map_path = os.path.join(run_path, "map")
    model_path = os.path.join(run_path, "model")
    os.makedirs(mesh_path, access, exist_ok=True)
    os.makedirs(map_path, access, exist_ok=True)
    os.makedirs(model_path, access, exist_ok=True)
    
    if config.wandb_vis_on:
        # set up wandb
        setup_wandb()
        wandb.init(project="SHINEMapping", config=vars(config), dir=run_path) # your own worksapce
        wandb.run.name = run_name         
    
    # o3d.utility.random.seed(42)

    return run_path


def setup_optimizer(config: SHINEConfig, octree_feat, mlp_param, sigma_size) -> Optimizer:
    lr_cur = config.lr
    opt_setting = []
    # weight_decay is for L2 regularization, only applied to MLP
    mlp_param_opt_dict = {'params': mlp_param, 'lr': lr_cur, 'weight_decay': config.weight_decay} 
    opt_setting.append(mlp_param_opt_dict)
    for i in range(config.tree_level_feat):
        # try to also add L2 regularization on the feature octree (results not quite good)
        feat_opt_dict = {'params': octree_feat[config.tree_level_feat-i-1], 'lr': lr_cur} 
        lr_cur *= config.lr_level_reduce_ratio
        opt_setting.append(feat_opt_dict)
    # make sigma also learnable for differentiable rendering (but not for our method)
    if config.ray_loss:
        sigma_opt_dict = {'params': sigma_size, 'lr': config.lr}
        opt_setting.append(sigma_opt_dict)
    
    if config.opt_adam:
        opt = optim.Adam(opt_setting, betas=(0.9,0.99), eps = config.adam_eps) 
    else:
        opt = optim.SGD(opt_setting, momentum=0.9)
    
    return opt
    

# set up weight and bias
def setup_wandb():
    print("Weight & Bias logging option is on. Disable it by setting  wandb_vis_on: False  in the config file.")
    username = getpass.getuser()
    # print(username)
    wandb_key_path = username + "_wandb.key"
    if not os.path.exists(wandb_key_path):
        wandb_key = input(
            "[You need to firstly setup and login wandb] Please enter your wandb key (https://wandb.ai/authorize):"
        )
        with open(wandb_key_path, "w") as fh:
            fh.write(wandb_key)
    else:
        print("wandb key already set")
    os.system('export WANDB_API_KEY=$(cat "' + wandb_key_path + '")')

def step_lr_decay(
    optimizer: Optimizer,
    learning_rate: float,
    iteration_number: int,
    steps: List,
    reduce: float = 1.0):

    if reduce > 1.0 or reduce <= 0.0:
        sys.exit(
            "The decay reta should be between 0 and 1."
        )

    if iteration_number in steps:
        steps.remove(iteration_number)
        learning_rate *= reduce
        print("Reduce base learning rate to {}".format(learning_rate))

        for param in optimizer.param_groups:
            param["lr"] *= reduce

    return learning_rate


def num_model_weights(model: nn.Module) -> int:
    num_weights = int(
        sum(
            [
                np.prod(p.size())
                for p in filter(lambda p: p.requires_grad, model.parameters())
            ]
        )
    )
    return num_weights


def print_model_summary(model: nn.Module):
    for child in model.children():
        print(child)


# coord, pred
def gradient(inputs, outputs):
    d_points = torch.ones_like(outputs, requires_grad=False, device=outputs.device)
    points_grad = grad(
        outputs=outputs,
        inputs=inputs,
        grad_outputs=d_points,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    return points_grad


def freeze_model(model: nn.Module):
    for child in model.children():
        for param in child.parameters():
            param.requires_grad = False


def unfreeze_model(model: nn.Module):
    for child in model.children():
        for param in child.parameters():
            param.requires_grad = True


def save_checkpoint(
    feature_octree, decoder_model, optimizer, run_path, checkpoint_name, iters
):
    torch.save(
        {
            "iters": iters,
            "feature_octree": feature_octree.state_dict(),
            "decoder": decoder_model.state_dict(),
            "optimizer": optimizer.state_dict(),
        },
        os.path.join(run_path, f"{checkpoint_name}.pth"),
    )
    print(f"save the model to {run_path}/{checkpoint_name}.pth")

def get_time():
    """
    :return: get timing statistics
    """
    torch.cuda.synchronize()
    return time.time()
