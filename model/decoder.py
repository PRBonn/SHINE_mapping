import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import grad

from utils.config import SHINEConfig


class Decoder(nn.Module):
    def __init__(self, config: SHINEConfig): 
        
        super().__init__()
        # predict sdf (now it anyway only predict sdf without further sigmoid
        # Initializa the structure of shared MLP
        layers = []
        for i in range(config.mlp_level):
            if i == 0:
                layers.append(nn.Linear(config.feature_dim, config.mlp_hidden_dim, config.mlp_bias_on))
            else:
                layers.append(nn.Linear(config.mlp_hidden_dim, config.mlp_hidden_dim, config.mlp_bias_on))
        self.layers = nn.ModuleList(layers)
        self.lout = nn.Linear(config.mlp_hidden_dim, 1, config.mlp_bias_on)
        # self.bn = nn.BatchNorm1d(self.hidden_dim, affine=False)

        self.to(config.device)
        # torch.cuda.empty_cache()

    def forward(self, feature):
        # If we use BCEwithLogits loss, do not need to do sigmoid mannually
        output = self.sdf(feature)
        return output

    # predict the sdf
    def sdf(self, sum_features):
        for k, l in enumerate(self.layers):
            if k == 0:
                h = F.relu(l(sum_features))
            else:
                h = F.relu(l(h))

        out = self.lout(h).squeeze(1)
        # linear (feature_dim -> hidden_dim)
        # relu
        # linear (hidden_dim -> hidden_dim)
        # relu
        # linear (hidden_dim -> 1)

        return out

    # predict the occupancy probability
    def occupancy(self, sum_features):
        out = torch.sigmoid(self.sdf(sum_features))  # to [0, 1]
        return out
