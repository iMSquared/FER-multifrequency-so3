import os
import sys
import copy
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

EPS = 1e-6


def knn(x: torch.Tensor, k: int) -> torch.Tensor:
    """Batched KNN

    Args:
        x (torch.Tensor): shape=[B, C, N]
        k (int): Number of neighbors

    Returns:
        idx (torch.Tensor): Indices of k-neighbers, shape=[B, N, K]
    """
    # Distance from every point to all other points.  
    inner = -2*torch.matmul(x.transpose(2, 1), x)           # [B, N, C] * [B, C, N] = [B, N, N]
    # |a|^2 for all points  
    xx = torch.sum(x**2, dim=1, keepdim=True)               # [B, 1, N]
    # Negative of pairwise distances, -|a-b|^2 = -a^2 - b^2 + 2ab.
    pairwise_distance = -xx - inner - xx.transpose(2, 1)    # [B, N, N]
    
    # Top k nearest
    idx = pairwise_distance.topk(k=k, dim=-1)[1]            # (B, N, K)
    return idx


def get_graph_feature(x: torch.Tensor, k: int = 20):
    """Aggregate graph feature from k nearest neighbors.
    
    Args:
        x (torch.Tensor): shape=[B, 1, 3, N]
        k (int, optional): Defaults to 20.

    Returns:
        torch.Tensor: Graph features, shape=[B, 2, 3, N, K]
            
    """
    batch_size = x.size(0)
    num_points = x.size(3)
    x = x.view(batch_size, -1, num_points)  # [B, 3, N]
    idx = knn(x, k=k)                       # [B, N, K]    
    device = x.device

    # This is used to flatten the batch...
    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1)*num_points # [B, 1, 1]
    idx = idx + idx_base                    # [B, N, K]
    idx = idx.view(-1)                      # [B*N*K]
 
    _, num_dims, _ = x.size()
    num_dims = num_dims // 3

    x = x.transpose(2, 1).contiguous()                              # [B, N, 3]    
    feature = x.view(batch_size*num_points, -1)[idx, :]             # [B*N*K, 3]        List up all neighbers of every point
    feature = feature.view(batch_size, num_points, k, num_dims, 3)  # [B, N, K, 1, 3]   3//3 is just like inserting a dim of 1...
    x = x.view(batch_size, num_points, 1, num_dims, 3)              # [B, N, 1, 1, 3]
    x = x.repeat(1, 1, k, 1, 1)                                     # [B, N, K, 1, 3]   Broadcast x to all its neigbors
    
    feature = torch.cat((feature-x, x), dim=3)                      # [B, N, K, 2, 3]   Local feat + global feat
    feature = feature.permute(0, 3, 4, 1, 2).contiguous()           # [B, 2, 3, N, K]   
  
    return feature


def get_graph_feature_cross(x: torch.Tensor, k: int = 20) -> torch.Tensor:
    """Aggregate graph feature from k nearest neighbors.

    Args:
        x (torch.Tensor): shape=[B, 1, 3, N]
        k (int, optional): Defaults to 20.

    Returns:
        torch.Tensor: Graph feature, shape=[B, 3, 3, N, K]
    """
    batch_size = x.size(0)
    num_points = x.size(3)
    x = x.view(batch_size, -1, num_points)  # [B, 3, N]
    idx = knn(x, k=k)                       # [B, N, K]
    device = torch.device(x.device)

    # This is used to flatten the batch...
    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1)*num_points # [B, 1, 1]
    idx = idx + idx_base                    # [B, N, K]
    idx = idx.view(-1)                      # [B*N*K]
 
    _, num_dims, _ = x.size()
    num_dims = num_dims

    x = x.transpose(2, 1).contiguous()                              # [B, N, 3]    
    feature = x.view(batch_size*num_points, -1)[idx, :]             # [B*N*K, 3]        List up all neighbers of every point
    feature = feature.view(batch_size, num_points, k, num_dims, 3)  # [B, N, K, 1, 3]   3//3 is just like inserting a dim of 1...
    x = x.view(batch_size, num_points, 1, num_dims, 3)              # [B, N, 1, 1, 3]
    x = x.repeat(1, 1, k, 1, 1)                                     # [B, N, K, 1, 3]   Broadcast x to all its neigbors
    cross = torch.cross(feature, x, dim=-1)                         # [B, N, K, 1, 3]   Cross product between x and its neighbors in R^3
    
    feature = torch.cat((feature-x, x, cross), dim=3)               # [B, N, K, 3, 3]   Local feat + global feat + cross feat
    feature = feature.permute(0, 3, 4, 1, 2).contiguous()           # [B, 3, 3, N, K]   
  
    return feature


class VNLinear(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(VNLinear, self).__init__()
        self.map_to_feat = nn.Linear(in_channels, out_channels, bias=False)
    
    def forward(self, x):
        '''
        x: point features of shape [B, N_feat, 3, N_samples, ...]
        '''
        x_out = self.map_to_feat(x.transpose(1,-1)).transpose(1,-1)
        return x_out


class VNLeakyReLU(nn.Module):
    def __init__(self, in_channels, share_nonlinearity=False, negative_slope=0.2):
        super(VNLeakyReLU, self).__init__()
        if share_nonlinearity == True:
            self.map_to_dir = nn.Linear(in_channels, 1, bias=False)
        else:
            self.map_to_dir = nn.Linear(in_channels, in_channels, bias=False)
        self.negative_slope = negative_slope
    
    def forward(self, x):
        '''
        x: point features of shape [B, N_feat, 3, N_samples, ...]
        '''
        d = self.map_to_dir(x.transpose(1,-1)).transpose(1,-1)
        dotprod = (x*d).sum(2, keepdim=True)
        mask = (dotprod >= 0).float()
        d_norm_sq = (d*d).sum(2, keepdim=True)
        x_out = self.negative_slope * x + (1-self.negative_slope) * (mask*x + (1-mask)*(x-(dotprod/(d_norm_sq+EPS))*d))
        return x_out


class VNLinearLeakyReLU(nn.Module):
    def __init__(self, in_channels, out_channels, dim=5, share_nonlinearity=False, use_batchnorm=True, negative_slope=0.2):
        super(VNLinearLeakyReLU, self).__init__()
        self.dim = dim
        self.share_nonlinearity = share_nonlinearity
        self.use_batchnorm = use_batchnorm
        self.negative_slope = negative_slope
        
        # Conv
        self.map_to_feat = nn.Linear(in_channels, out_channels, bias=False)
        
        # BatchNorm
        self.use_batchnorm = use_batchnorm
        if use_batchnorm == True:
            self.batchnorm = VNBatchNorm(out_channels, dim=dim)
        
        # LeakyReLU
        if share_nonlinearity == True:
            self.map_to_dir = nn.Linear(in_channels, 1, bias=False)
        else:
            self.map_to_dir = nn.Linear(in_channels, out_channels, bias=False)
        self.negative_slope = negative_slope
    
    def forward(self, x):
        '''
        x: point features of shape [B, N_feat, 3, N_samples, ...]
        '''
        # Conv
        p = self.map_to_feat(x.transpose(1,-1)).transpose(1,-1)
        # InstanceNorm
        if self.use_batchnorm == True:
            p = self.batchnorm(p)
        # LeakyReLU
        d = self.map_to_dir(x.transpose(1,-1)).transpose(1,-1)
        dotprod = (p*d).sum(2, keepdim=True)
        mask = (dotprod >= 0).float()
        d_norm_sq = (d*d).sum(2, keepdim=True)
        x_out = self.negative_slope * p + (1-self.negative_slope) * (mask*p + (1-mask)*(p-(dotprod/(d_norm_sq+EPS))*d))
        return x_out


class VNBatchNorm(nn.Module):
    def __init__(self, num_features, dim):
        super(VNBatchNorm, self).__init__()
        self.dim = dim
        if dim == 3 or dim == 4:
            self.bn = nn.BatchNorm1d(num_features)
        elif dim == 5:
            self.bn = nn.BatchNorm2d(num_features)
    
    def forward(self, x):
        '''
        x: point features of shape [B, N_feat, 3, N_samples, ...]
        '''
        norm = torch.sqrt((x*x).sum(2))
        norm_bn = self.bn(norm)
        norm = norm.unsqueeze(2)
        norm_bn = norm_bn.unsqueeze(2)
        x = x / norm * norm_bn
        
        return x


class VNMaxPool(nn.Module):
    def __init__(self, in_channels, share_nonlinearity=False):
        super(VNMaxPool, self).__init__()
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


class VNStdFeature(nn.Module):
    def __init__(self, in_channels, dim=4, normalize_frame=False, share_nonlinearity=False, use_batchnorm=True):
        super(VNStdFeature, self).__init__()
        self.dim = dim
        self.normalize_frame = normalize_frame
        self.share_nonlinearity = share_nonlinearity
        self.use_batchnorm = use_batchnorm
        
        self.vn1 = VNLinearLeakyReLU(in_channels, in_channels//2, dim=dim, share_nonlinearity=share_nonlinearity, use_batchnorm=use_batchnorm)
        self.vn2 = VNLinearLeakyReLU(in_channels//2, in_channels//4, dim=dim, share_nonlinearity=share_nonlinearity, use_batchnorm=use_batchnorm)
        if normalize_frame:
            self.vn_lin = nn.Linear(in_channels//4, 2, bias=False)
        else:
            self.vn_lin = nn.Linear(in_channels//4, 3, bias=False)
    
    def forward(self, x):
        '''
        x: point features of shape [B, N_feat, 3, N_samples, ...]
        '''
        z0 = self.vn1(x)
        z0 = self.vn2(z0)
        z0 = self.vn_lin(z0.transpose(1, -1)).transpose(1, -1)
        
        if self.normalize_frame:
            # make z0 orthogonal. u2 = v2 - proj_u1(v2)
            v1 = z0[:,0,:]
            #u1 = F.normalize(v1, dim=1)
            v1_norm = torch.sqrt((v1*v1).sum(1, keepdim=True))
            u1 = v1 / (v1_norm+EPS)
            v2 = z0[:,1,:]
            v2 = v2 - (v2*u1).sum(1, keepdim=True)*u1
            #u2 = F.normalize(u2, dim=1)
            v2_norm = torch.sqrt((v2*v2).sum(1, keepdim=True))
            u2 = v2 / (v2_norm+EPS)

            # compute the cross product of the two output vectors       
            u3 = torch.cross(u1, u2)
            z0 = torch.stack([u1, u2, u3], dim=1).transpose(1, 2)
        else:
            z0 = z0.transpose(1, 2)
        
        if self.dim == 4:
            x_std = torch.einsum('bijm,bjkm->bikm', x, z0)
        elif self.dim == 3:
            x_std = torch.einsum('bij,bjk->bik', x, z0)
        elif self.dim == 5:
            x_std = torch.einsum('bijmn,bjkmn->bikmn', x, z0)
        
        return x_std, z0


# Resnet Blocks
class VNResnetBlockFC(nn.Module):
    ''' Fully connected ResNet Block class.

    Args:
        size_in (int): input dimension
        size_out (int): output dimension
        size_h (int): hidden dimension
    '''

    def __init__(self, size_in, size_out=None, size_h=None):
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
        self.fc_0 = VNLinear(size_in, size_h)
        self.fc_1 = VNLinear(size_h, size_out)
        self.actvn_0 = VNLeakyReLU(size_in, negative_slope=0.0, share_nonlinearity=False)
        self.actvn_1 = VNLeakyReLU(size_h, negative_slope=0.0, share_nonlinearity=False)

        if size_in == size_out:
            self.shortcut = None
        else:
            self.shortcut = VNLinear(size_in, size_out)
        # Initialization
        nn.init.zeros_(self.fc_1.map_to_feat.weight)

    def forward(self, x):
        net = self.fc_0(self.actvn_0(x))
        dx = self.fc_1(self.actvn_1(net))

        if self.shortcut is not None:
            x_s = self.shortcut(x)
        else:
            x_s = x

        return x_s + dx
