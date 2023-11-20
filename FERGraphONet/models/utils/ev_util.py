"""
Modified from https://github.com/FlyingGiraffe/vnn-pc
"""

import numpy as np
import torch
import einops

from models.utils.rotm_util import Y_func_V2

EPS = 1e-6

def knn(x, k):
    inner = -2*torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x**2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
 
    idx = pairwise_distance.topk(k=k, dim=-1)[1]   # (batch_size, num_points, k)
    return idx


def get_graph_feature_ext(x: torch.Tensor, x_ext=None, k: int = 20) -> torch.Tensor: 
    """_summary_

    Args:
        x (torch.Tensor): shape=[B, 1, D, N]
        k (int, optional): Number of neighbers. Defaults to 20.
        idx (_type_, optional): ????. Defaults to None.

    Returns:
        _type_: _description_
    """
    batch_size = x.size(0)
    num_points = x.size(3)
    x = x.view(batch_size, -1, num_points) # [B, 1*D, N]
    if x_ext is None:
        x_ext = x
    feat_dim = x_ext.size(1)
    idx = knn(x, k=k)
    device = x.device

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1)*num_points

    idx = idx + idx_base

    idx = idx.view(-1)
 
    _, num_dims, _ = x_ext.size()
    num_dims = num_dims // feat_dim

    x = x.transpose(2, 1).contiguous()
    x_ext = x_ext.transpose(2, 1).contiguous()
    feature = x_ext.view(batch_size*num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims, feat_dim) 
    x_ext = x_ext.view(batch_size, num_points, 1, num_dims, feat_dim).repeat(1, 1, k, 1, 1)
    feature = torch.cat((feature-x_ext, x_ext), dim=3).permute(0, 3, 4, 1, 2).contiguous()
  
    return feature

def get_graph_feature(x, k=20, idx=None, x_coord=None):
    batch_size = x.size(0)
    num_points = x.size(3)
    feat_dim = x.size(2)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        if x_coord is None: # dynamic knn graph
            idx = knn(x, k=k)   # (batch_size, num_points, k)
        else:          # fixed knn graph with input point coordinates
            x_coord = x_coord.view(batch_size, -1, num_points)
            idx = knn(x_coord, k=k)
    device = torch.device('cuda')

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1)*num_points

    idx = idx + idx_base

    idx = idx.view(-1)
 
    _, num_dims, _ = x.size()
    num_dims = num_dims // feat_dim

    x = x.transpose(2, 1).contiguous()   # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
    feature = x.view(batch_size*num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims, feat_dim) 
    x = x.view(batch_size, num_points, 1, num_dims, feat_dim).repeat(1, 1, k, 1, 1)
    
    feature = torch.cat((feature-x, x), dim=3).permute(0, 3, 4, 1, 2).contiguous()
  
    return feature



def get_graph_feature_cross(x, k=20, idx=None):
    batch_size = x.size(0)
    num_points = x.size(3)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        idx = knn(x, k=k)

    idx_base = torch.arange(0, batch_size, device=x.device).view(-1, 1, 1)*num_points

    idx = idx + idx_base

    idx = idx.view(-1)
 
    _, num_dims, _ = x.size()
    num_dims = num_dims // 3

    x = x.transpose(2, 1).contiguous()
    feature = x.view(batch_size*num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims, 3) 
    x = x.view(batch_size, num_points, 1, num_dims, 3).repeat(1, 1, k, 1, 1)
    cross = torch.cross(feature, x, dim=-1)
    
    feature = torch.cat((feature-x, x, cross), dim=3).permute(0, 3, 4, 1, 2).contiguous()
  
    return feature

def in_normalization_features(x, dim_list, scale_params, feat_axis=-2):
    '''
    feature_dim
    '''
    x = torch.transpose(x, feat_axis, -1)
    ft_list = []
    sidx = 0
    for i, dl in enumerate(dim_list):
        n = dl*2+1
        cur_x = x[...,sidx:sidx+n]
        cur_x_norm = cur_x.norm(dim=-1, keepdim=True)
        cur_x = cur_x/cur_x_norm.mean(dim=tuple(np.arange(1, len(x.size())-1, dtype=np.int)), keepdim=True)
        if scale_params is not None:
            cur_x = cur_x*scale_params[i]
        ft_list.append(cur_x)
        sidx = sidx+n
    x = torch.concat(ft_list, dim=-1)
    return torch.transpose(x, feat_axis, -1).contiguous()


def make_features(x, rot_configs, scale_params, feat_axis=-2):
    '''
    '''
    x = torch.transpose(x, feat_axis, -1)
    ft_list = []
    for i, dl in enumerate(rot_configs['dim_list']):
        x_ = Y_func_V2(dl, x, rot_configs)
        x_ = torch.transpose(x_, feat_axis, -1)
        if scale_params is not None:
            x_ = x_ * scale_params[i]
        ft_list.append(x_)

    return torch.concat(ft_list, feat_axis).contiguous()


def get_feat(x, l, rot_configs, feature_axis=-2):
    idx = rot_configs['dim_list'].index(l)
    sidx = np.sum(np.array(rot_configs['dim_list'][:idx]) * 2 + 1).astype(int)
    eidx = sidx + 2*l+1
    return x.transpose(feature_axis, -1)[..., sidx:eidx].transpose(feature_axis, -1)

def contract_to_3d(x1, l1, x2, l2, rot_configs, feature_axis=-2):
    
    x1 = torch.transpose(x1, feature_axis, -1)
    x2 = torch.transpose(x2, feature_axis, -1)

    x12 = einops.rearrange(torch.einsum('...i,...j', x1, x2), '... i j->... (i j)')
    x3d_ = torch.einsum('ij,...j', rot_configs['Qs'][(l1, l2)], x12)
    x3d_ = x3d_.transpose(feature_axis, -1).contiguous()
    return x3d_