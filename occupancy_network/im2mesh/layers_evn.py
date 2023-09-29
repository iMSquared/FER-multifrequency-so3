import os
import sys
import copy
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict
from pathlib import Path

from im2mesh.layers_vnn import knn, get_graph_feature_cross

import rotm_util_common as rmutil


EPS = 1e-6


class MakeHDFeature(nn.Module):
    def __init__(self, rot_configs, nfeat=2,feature_axis=2):
        super(MakeHDFeature, self).__init__()
        self.rot_configs = rot_configs
        self.radial_func = nn.ModuleList()
        self.feature_axis = feature_axis
        for _ in self.rot_configs['dim_list']:
            self.radial_func.append(nn.Sequential(
                nn.Linear(nfeat,16),
                nn.LeakyReLU(0.2),
                nn.Linear(16,16),
                nn.LeakyReLU(0.2),
                nn.Linear(16,nfeat),
            ))

    def forward(self, x):
        x = torch.transpose(x, self.feature_axis, -1)
        x_norm = x.norm(dim=-1, keepdim=True) + EPS
        x_hat = x/x_norm
        ft_list = []
        for i, dl in enumerate(self.rot_configs['dim_list']):
            x_ = rmutil.Y_func_V2(dl, x_hat, self.rot_configs)
            x_norm_ = self.radial_func[i](x_norm.transpose(1,-1)).transpose(1,-1)
            x_ = x_*x_norm_
            ft_list.append(x_)
        feat = torch.concat(ft_list, -1)
        return feat.transpose(self.feature_axis, -1).contiguous()



def get_graph_feature_evn(x: torch.Tensor, x_ext: torch.Tensor, k: int = 20):
    """Aggregate graph feature from k nearest neighbors.

    Args:
        x (torch.Tensor): Find KNN in x. shape=[B, 1, 3, N]
        x_ext (torch.Tensor): Extract feature from x_ext. shape=[B, 1, R, N]
        k (int, optional): Defaults to 20.

    Returns:
        torch.Tensor: Graph feature, shape=[B, 2, R, N, K]
    """
    batch_size = x_ext.size(0)  # B
    feat_dim   = x_ext.size(2)  # R
    num_points = x_ext.size(3)  # N
    # KNN must be done in R^3 space
    x     = x.view(batch_size, -1, num_points)      # [B, R, N]
    x_ext = x_ext.view(batch_size, -1, num_points)  # [B, R, N]
    idx = knn(x, k=k)                               # [B, N, K]
    device = torch.device(x.device)

    # This is used to flatten the batch...
    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1)*num_points # [B, 1, 1]
    idx = idx + idx_base                    # [B, N, K]
    idx = idx.view(-1)                      # [B*N*K]
 
    # Reformat
    x_ext = x_ext.transpose(2, 1).contiguous()                      # [B, N, R]    
    feature = x_ext.view(batch_size*num_points, -1)[idx, :]         # [B*N*K, R]        List up all neighbers of every point
    feature = feature.view(batch_size, num_points, k, 1, feat_dim)  # [B, N, K, 1, R]   
    x_ext = x_ext.view(batch_size, num_points, 1, 1, feat_dim)      # [B, N, 1, 1, R]
    x_ext = x_ext.repeat(1, 1, k, 1, 1)                             # [B, N, K, 1, R]   Broadcast x to all its neigbors
    # Aggregate
    feature = torch.cat((feature-x_ext, x_ext), dim=3)              # [B, N, K, 2, R]   Local feat + global feat
    feature = feature.permute(0, 3, 4, 1, 2).contiguous()           # [B, 2, R, N, K]   Features repeated by K times.
  
    return feature


def get_graph_feature_cross_evn(x: torch.Tensor, x_ext: torch.Tensor, k: int = 20) -> torch.Tensor:
    """Aggregate graph feature from k nearest neighbors.

    Args:
        x (torch.Tensor): Find KNN in x. shape=[B, 1, 3, N]
        x_ext (torch.Tensor): Extract feature from x_ext. shape=[B, 1, R, N]
        k (int, optional): Defaults to 20.

    Returns:
        torch.Tensor: Graph feature, shape=[B, 3, R, N, K]
    """
    batch_size = x_ext.size(0)  # B
    feat_dim   = x_ext.size(2)  # R
    num_points = x_ext.size(3)  # N
    # KNN must be done in R^3 space
    x     = x.view(batch_size, -1, num_points)      # [B, R, N]
    x_ext = x_ext.view(batch_size, -1, num_points)  # [B, R, N]
    idx = knn(x, k=k)                               # [B, N, K]
    device = torch.device(x.device)

    # This is used to find neighbor in flattened batch...
    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1)*num_points # [B, 1, 1]
    idx = idx + idx_base                    # [B, N, K]
    idx = idx.view(-1)                      # [B*N*K]
 
    # Reformat
    x_ext = x_ext.transpose(2, 1).contiguous()                      # [B, N, R]    
    feature = x_ext.view(batch_size*num_points, -1)[idx, :]         # [B*N*K, R]        List up all neighbers of every point
    feature = feature.view(batch_size, num_points, k, 1, feat_dim)  # [B, N, K, 1, R]   
    x_ext = x_ext.view(batch_size, num_points, 1, 1, feat_dim)      # [B, N, 1, 1, R]
    x_ext = x_ext.repeat(1, 1, k, 1, 1)                             # [B, N, K, 1, R]   Broadcast x to all its neigbors
    # Cross (Failed if not R^3)
    cross = torch.cross(feature, x_ext, dim=-1)                     # [B, N, K, 1, R]   Cross product between x and its neighbors in R^3
    # Aggregate
    feature = torch.cat((feature-x_ext, x_ext, cross), dim=3)       # [B, N, K, 3, R]   Local feat + global feat + cross feat
    feature = feature.permute(0, 3, 4, 1, 2).contiguous()           # [B, 3, R, N, K]   Features repeated by K times.
  
    return feature


def make_features(x: torch.Tensor, 
                  rot_configs: Dict, 
                  scale_params: torch.nn.Parameter):
    """EVN: Map input R^3 pointcloud -> {R^n, ...} pointcloud
    
    Args:
        x (torch.Tensor): shape=[..., 3, N]
        rot_configs (Dict): Check out rototation util.
        scale_params (torch.nn.Parameter): Scale parameters

    Returns:
        torch.Tensor: shape=[..., R, N]
    """
    x = torch.transpose(x, -2, -1)
    ft_list = []
    for i, dl in enumerate(rot_configs['dim_list']):
        x_ = rmutil.Y_func_V2(dl, x, rot_configs)
        x_ = torch.transpose(x_, -2, -1)
        if i!= 0 and scale_params is not None:
            x_ = x_*scale_params[i-1]
        ft_list.append(x_)

    return torch.concat(ft_list, -2)


def in_normalization_features(x: torch.Tensor, 
                              rot_configs: Dict, 
                              scale_params: torch.nn.Parameter):
    """EVN: ???
    
    Args:
        x (torch.Tensor): shape=
        rot_configs (Dict): _
        scale_params (torch.nn.Parameter): _
    
    Returns:
        torch.Tensor: shape=x.shape
    """
    if rot_configs is None:
        return x
    x = torch.transpose(x, -2, -1)
    ft_list = []
    sidx = 0
    for i, dl in enumerate(rot_configs['dim_list']):
        n = dl*2+1
        cur_x = x[...,sidx:sidx+n]
        cur_x_norm = cur_x.norm(dim=-1, keepdim=True)
        cur_x = cur_x/cur_x_norm.mean(dim=(-2,-3), keepdim=True)
        if scale_params is not None:
            cur_x = cur_x * scale_params[i]
        ft_list.append(cur_x)
        sidx = sidx+n
    x = torch.concat(ft_list, dim=-1)
    return torch.transpose(x, -2, -1).contiguous()


class EVNLinear(nn.Module):
    """Same as VN"""
    def __init__(self, in_channels: int, 
                       out_channels: int, 
                       rot_configs: Dict = None):
        super(EVNLinear, self).__init__()
        # Debug
        assert rot_configs is not None

        self.rot_configs = rot_configs
        self.map_to_feat = nn.Linear(in_channels, out_channels, bias=False)
    
    def forward(self, x):
        '''
        x: point features of shape [B, N_feat, 3, N_samples, ...]
        '''
        x_out = self.map_to_feat(x.transpose(1,-1)).transpose(1,-1)
        return x_out


class EVNLeakyReLU(nn.Module):
    """EVN: This module applies the nonlinearity in each sub-feature space."""
    def __init__(self, in_channels: int, 
                       share_nonlinearity: bool = False, 
                       negative_slope: bool = 0.2, 
                       rot_configs: Dict = None):
        super(EVNLeakyReLU, self).__init__()
        # Debug
        assert rot_configs is not None

        if share_nonlinearity == True:
            self.map_to_dir = nn.Linear(in_channels, 1, bias=False)
        else:
            self.map_to_dir = nn.Linear(in_channels, in_channels, bias=False)
        self.negative_slope = negative_slope
        # EVN
        self.rot_configs = rot_configs
    
    def forward(self, x):
        '''
        x: point features of shape [B, N_feat, 3, N_samples, ...]
        '''
        d = self.map_to_dir(x.transpose(1,-1)).transpose(1,-1)  
        # VN
        if self.rot_configs is None:
            dotprod = (x*d).sum(2, keepdim=True)
            mask = (dotprod >= 0).float()
            d_norm_sq = (d*d).sum(2, keepdim=True)
            x_out = self.negative_slope * x + (1-self.negative_slope) * (mask*x + (1-mask)*(x-(dotprod/(d_norm_sq+EPS))*d))
        # EVN
        else:
            x_out = []
            sidx = 0
            for rd in self.rot_configs['dim_list']:
                # Take sub-feature
                n = rd*2+1
                x_ = x[:,:,sidx:sidx+n]
                d_ = d[:,:,sidx:sidx+n]
                # Sub-feature-wise nonlinearity.
                dotprod = (x_*d_).sum(2, keepdims=True)
                mask = (dotprod >= 0).float()
                d_norm_sq = (d_*d_).sum(2, keepdims=True)
                x_out_ = self.negative_slope * x_ + (1-self.negative_slope) * (mask*x_ + (1-mask)*(x_-(dotprod/(d_norm_sq+EPS))*d_))
                x_out.append(x_out_)
                sidx += n
            x_out = torch.concat(x_out, dim=2).contiguous()

        return x_out


class EVNLinearLeakyReLU(nn.Module):
    """EVN: This module applies the nonlinearity in each sub-feature space."""
    def __init__(self, in_channels: int, 
                       out_channels: int, 
                       dim: int = 5, 
                       share_nonlinearity: bool = False, 
                       negative_slope: float = 0.2, 
                       rot_configs: Dict = None):
        super(EVNLinearLeakyReLU, self).__init__()
        # Debug
        assert rot_configs is not None

        self.dim = dim
        # Conv
        self.map_to_feat = nn.Linear(in_channels, out_channels, bias=False)        
        # LeakyReLU
        self.share_nonlinearity = share_nonlinearity
        if share_nonlinearity == True:
            self.map_to_dir = nn.Linear(in_channels, 1, bias=False)
        else:
            self.map_to_dir = nn.Linear(in_channels, out_channels, bias=False)
        self.negative_slope = negative_slope
        # EVN
        self.rot_configs = rot_configs
    
    def forward(self, x):
        '''
        x: point features of shape [B, N_feat, 3, N_samples, ...]
        '''
        # Linear
        p = self.map_to_feat(x.transpose(1,-1)).transpose(1,-1)
        # LeakyReLU
        d = self.map_to_dir(x.transpose(1,-1)).transpose(1,-1)
        # VN
        if self.rot_configs is None:
            dotprod = (p*d).sum(2, keepdim=True)
            mask = (dotprod >= 0).float()
            d_norm_sq = (d*d).sum(2, keepdim=True)
            x_out = self.negative_slope * p + (1-self.negative_slope) * (mask*p + (1-mask)*(p-(dotprod/(d_norm_sq+EPS))*d))
        # EVN
        else:
            x_out = []
            sidx = 0
            for rd in self.rot_configs['dim_list']:
                # Take sub-feature
                n = rd*2+1
                p_ = p[:,:,sidx:sidx+n]
                d_ = d[:,:,sidx:sidx+n]
                # Sub-feature-wise nonlinearity.
                dotprod = (p_*d_).sum(2, keepdims=True)
                mask = (dotprod >= 0).float()
                d_norm_sq = (d_*d_).sum(2, keepdims=True)
                x_out_ = self.negative_slope * p_ + (1-self.negative_slope) * (mask*p_ + (1-mask)*(p_-(dotprod/(d_norm_sq+EPS))*d_))
                x_out.append(x_out_)
                sidx += n
            x_out = torch.concat(x_out, dim=2).contiguous()

        return x_out
    

class EVNLinearMagNL(nn.Module):
    """EVN: in progress..."""
    def __init__(self, in_channels: int, 
                       out_channels: int, 
                       dim: int = 5, 
                       rot_configs: Dict = None):
        super(EVNLinearMagNL, self).__init__()
        # Debug
        assert rot_configs is not None

        self.dim = dim
        # Conv
        self.map_to_feat = nn.Linear(in_channels, out_channels, bias=False)        
        self.actn = EVNMagNL(out_channels, rot_configs)
        self.rot_configs = rot_configs
    
    def forward(self, x):
        '''
        x: point features of shape [B, N_feat, 3, N_samples, ...]
        '''
        # Linear
        p = self.map_to_feat(x.transpose(1,-1)).transpose(1,-1)
        x_out = self.actn(p)

        return x_out


class EVNMagNL(nn.Module):
    """EVN: Novel nonlinearity!!!"""
    def __init__(self, num_features, rot_configs):
        super(EVNMagNL, self).__init__()
        self.gate = nn.Sequential(
            nn.Linear(num_features, num_features//2, bias=True),
            nn.ReLU(),
            nn.Linear(num_features//2, num_features, bias=True),)
    
    def forward(self, x):
        '''
        x: point features of shape [B, N_feat, 3, N_samples, ...]
        '''
        # norm = torch.sqrt((x*x).sum(2))
        norm = torch.norm(x, dim=2) + EPS   # [B, F, N]
        norm_bn = self.gate(norm.transpose(1,-1)).transpose(1,-1)   # [B, F, N]
        norm = norm.unsqueeze(2)            # [B, F, 1, N]
        norm_bn = norm_bn.unsqueeze(2)      # [B, F, 1, N]
        x = x / norm * norm_bn
        
        return x


class EVNMaxPool(nn.Module):
    """Same as VN"""
    def __init__(self, in_channels, share_nonlinearity=False, rot_configs: Dict = None):
        super(EVNMaxPool, self).__init__()
        # Debug
        assert rot_configs is not None

        if share_nonlinearity:
            self.map_to_dir = nn.Linear(in_channels, 1, bias=False)
        else:
            self.map_to_dir = nn.Linear(in_channels, in_channels, bias=False)
    
    def forward(self, x):
        '''
        x: point features of shape [B, N_feat, 3, N_samples, ...]
        '''
        d = self.map_to_dir(x.transpose(1,-1)).transpose(1,-1)
        dotprod = (x*d).sum(2, keepdim=True)
        idx = dotprod.max(dim=-1, keepdim=False)[1]
        index_tuple = torch.meshgrid([torch.arange(j) for j in x.size()[:-1]]) + (idx,)
        x_max = x[index_tuple]
        return x_max


# Resnet Blocks
class EVNResnetBlockFC(nn.Module):
    ''' Fully connected ResNet Block class.

    Args:
        size_in (int): input dimension
        size_out (int): output dimension
        size_h (int): hidden dimension
    '''

    def __init__(self, size_in: int, 
                       size_out: int = None, 
                       size_h: int = None,
                       rot_configs: Dict = None):
        super().__init__()
        # Attributes
        if size_out is None:
            size_out = size_in
        if size_h is None:
            size_h = min(size_in, size_out)
        self.size_in = size_in
        self.size_h = size_h
        self.size_out = size_out
        # Submodules
        self.fc_0 = EVNLinear(size_in, size_h, rot_configs)
        self.fc_1 = EVNLinear(size_h, size_out, rot_configs)
        self.actvn_0 = EVNMagNL(size_in, rot_configs)
        self.actvn_1 = EVNMagNL(size_h, rot_configs)

        if size_in == size_out:
            self.shortcut = None
        else:
            self.shortcut = EVNLinear(size_in, size_out, rot_configs)
        # Initialization
        nn.init.zeros_(self.fc_1.map_to_feat.weight)
        # EVN
        self.rot_configs = rot_configs


    def forward(self, x):
        net = self.fc_0(self.actvn_0(x))
        dx = self.fc_1(self.actvn_1(net))

        if self.shortcut is not None:
            x_s = self.shortcut(x)
        else:
            x_s = x

        return x_s + dx
