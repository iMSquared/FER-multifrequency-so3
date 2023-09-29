import torch
import torch.nn as nn
from im2mesh.layers_evn import *

from typing import List, Dict
from dataclasses import dataclass, field


def maxpool(x, dim=-1, keepdim=False):
    out, _ = x.max(dim=dim, keepdim=keepdim)
    return out


def meanpool(x, dim=-1, keepdim=False):
    out = x.mean(dim=dim, keepdim=keepdim)
    return out


class EVN_ResnetPointnet(nn.Module):
    """ PointNet-based EVN encoder network with ResNet block.

    Args:
        c_dim (int): dimension of latent code c
        dim (int): input points dimension
        hidden_dim (int): hidden dimension of the network
        k (int): number of neighbers at input
        use_scale_params (bool): _
        rot_config (EVNConfig): Some configurations...
        use_cross_feat (bool): Use cross product graph feature
        
    """
    
    def __init__(self, c_dim: int = 128, 
                       dim: int = 3, 
                       hidden_dim: int = 128, 
                       k: int = 20, 
                       use_scale_params: bool = False,
                       use_radial: bool = False,
                       rot_configs: Dict = None):
        super().__init__()
        self.c_dim       = c_dim
        self.k           = k
        # Debug
        assert rot_configs is not None
        # EVN
        self.use_scale_params = use_scale_params
        self.use_radial       = use_radial
        self.rot_configs      = rot_configs

        # EVN: New feature
        if use_radial:
            self.make_features = MakeHDFeature(rot_configs, 3, 2)
            in_feat_dim = 3
        else:
            in_feat_dim = 2

        # EVN: Adjust scales at each feature spaces.
        if self.use_scale_params:
            num_rot_spaces = len(self.rot_configs["dim_list"])
            self.scale_params = nn.Parameter(
                data          = torch.Tensor(torch.ones(num_rot_spaces)),
                requires_grad = True)
        
        # Layers
        self.conv_pos = EVNLinearMagNL(in_feat_dim, 128, rot_configs=rot_configs)
        self.fc_pos = EVNLinear(128, 2*hidden_dim, rot_configs=rot_configs)
        self.block_0 = EVNResnetBlockFC(2*hidden_dim, hidden_dim, rot_configs=rot_configs)
        self.block_1 = EVNResnetBlockFC(2*hidden_dim, hidden_dim, rot_configs=rot_configs)
        self.block_2 = EVNResnetBlockFC(2*hidden_dim, hidden_dim, rot_configs=rot_configs)
        self.block_3 = EVNResnetBlockFC(2*hidden_dim, hidden_dim, rot_configs=rot_configs)
        self.block_4 = EVNResnetBlockFC(2*hidden_dim, hidden_dim, rot_configs=rot_configs)
        self.fc_c = EVNLinear(hidden_dim, c_dim, rot_configs=rot_configs)

        self.actvn_c = EVNMagNL(hidden_dim, rot_configs)
        self.pool = meanpool


    def forward(self, p):
        """
        Args:
            p: [B, N, 3]

        Returns:
            torch.Tensor: [B, c_dim]
        """
        batch_size = p.size(0)              # [B, N, 3]
        p = p.unsqueeze(1).transpose(2, 3)  # [B, 1, 3, N]

        if self.use_radial:
            feat = get_graph_feature_cross(p, k=self.k)   # [B, 3, 3, N, K]
            feat = self.make_features(feat)               # [B, 3, R, N, K]
        else:
            if self.use_scale_params:
                p_ext = make_features(p, self.rot_configs, self.scale_params)   # [B, 1, R, N]
            else:
                p_ext = make_features(p, self.rot_configs, None)   # [B, 1, R, N]
            feat = get_graph_feature_evn(p, p_ext, k=self.k)       # [B, 2, R, N, K]

        net = self.conv_pos(feat)                                   # [B, 128, R, N, K]
        net = self.pool(net, dim=-1)                                # [B, 128, R, N]

        net = self.fc_pos(net)                                      # [B, 2F, R, N]

        net = self.block_0(net)                                     # [B, F, R, N]
        pooled = self.pool(net, dim=-1, keepdim=True).expand(net.size())
        net = torch.cat([net, pooled], dim=1)
        
        net = self.block_1(net)                                     # [B, F, R, N]
        pooled = self.pool(net, dim=-1, keepdim=True).expand(net.size())
        net = torch.cat([net, pooled], dim=1)
        
        net = self.block_2(net)                                     # [B, F, R, N]
        pooled = self.pool(net, dim=-1, keepdim=True).expand(net.size())
        net = torch.cat([net, pooled], dim=1)
        
        net = self.block_3(net)                                     # [B, F, R, N]
        pooled = self.pool(net, dim=-1, keepdim=True).expand(net.size())
        net = torch.cat([net, pooled], dim=1)

        net = self.block_4(net)                                     # [B, F, R, N]

        # Reduce to [B, 128, R]
        net = self.pool(net, dim=-1)

        c = self.fc_c(self.actvn_c(net))    # [B, c_dim, R]

        return c