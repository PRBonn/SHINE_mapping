import math
from tqdm import tqdm
from model.feature_octree import FeatureOctree
from model.decoder import Decoder
from dataset.lidar_dataset import LiDARDataset
from utils.loss import *

def cal_feature_importance(data: LiDARDataset, octree: FeatureOctree, mlp: Decoder, 
    sigma, bs, down_rate=1, loss_reduction='mean', loss_weight_on = False):
    
    # shuffle_indice = torch.randperm(data.coord_pool.shape[0])
    # shuffle_coord = data.coord_pool[shuffle_indice]
    # shuffle_label = data.sdf_label_pool[shuffle_indice]

    sample_count = data.coord_pool.shape[0]
    batch_interval = bs*down_rate
    iter_n = math.ceil(sample_count/batch_interval)
    for n in tqdm(range(iter_n)):
        head = n*batch_interval
        tail = min((n+1)*batch_interval, sample_count)
        # batch_coord = data.coord_pool[head:tail:down_rate]
        # batch_label = data.sdf_label_pool[head:tail:down_rate]

        batch_coord = data.coord_pool[head:tail:down_rate]
        batch_label = data.sdf_label_pool[head:tail:down_rate]
        # batch_weight = data.weight_pool[head:tail:down_rate]
        count = batch_label.shape[0]
            
        octree.get_indices(batch_coord)
        features = octree.query_feature(batch_coord)
        pred = mlp(features) # before sigmoid         
        # add options for other losses here                              
        sdf_loss = sdf_bce_loss(pred, batch_label, sigma, None, loss_weight_on, loss_reduction)                         
        sdf_loss.backward()

        for i in range(len(octree.importance_weight)): # for each level
            octree.importance_weight[i] += octree.hier_features[i].grad.abs()
            octree.hier_features[i].grad.zero_()
        
            octree.importance_weight[i][-1] *= 0 # reseting the trashbin feature weight to 0 