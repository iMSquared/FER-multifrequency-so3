import os
import sys
import copy
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import models.ev_layers as evl

EPS = 1e-8
EPS2 = 1e-12

def knn(x, k):
    with torch.no_grad():
        inner = -2 * torch.matmul(x.transpose(2, 1), x)
        xx = torch.sum(x ** 2, dim=1, keepdim=True)
        pairwise_distance = -xx - inner - xx.transpose(2, 1)

        idx = pairwise_distance.topk(k=k, dim=-1)[1]  # (batch_size, num_points, k)
        return idx

def knn2(x, y, k):
    '''

    :param x: source points
    :param y: target points
    :param k:
    :return:
    '''
    with torch.no_grad():
        inner = -2 * torch.matmul(x.transpose(2, 1), y)
        # xy = - 2 * torch.matmul(x.transpose(2, 1), y)
        # yx = - torch.matmul(y.transpose(2, 1), x)
        xx = torch.sum(x ** 2, dim=1, keepdim=True)
        yy = torch.sum(y ** 2, dim=1, keepdim=True)
        pairwise_distance = -xx.transpose(2, 1) - inner - yy

        idx = pairwise_distance.topk(k=k, dim=-1)[1]  # (batch_size, num_points, k)
        return idx

def vn_get_graph_feature(x, k=20, idx=None, x_coord=None):
    batch_size, num_dims, num_feat, num_points = x.shape  # B,1,3,N

    x = x.view(batch_size, -1, num_points)
    if idx is None:
        if x_coord is None:  # dynamic knn graph
            idx = knn(x, k=k)
        else:  # fixed knn graph with input point coordinates
            idx = knn(x_coord, k=k)
    device = torch.device('cuda')
    try:
        idx_base = torch.arange(0, batch_size, device=device) * num_points
    except:
        print(batch_size)
    idx_base.unsqueeze_(1).unsqueeze_(2)

    idx = idx + idx_base

    idx = idx.view(-1).contiguous()

    #_, num_dims, _ = x.size()
    #num_dims = num_dims // 3

    x = x.transpose(2, 1).contiguous()
    feature = x.view(batch_size * num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims, 3)
    x = x.view(batch_size, num_points, 1, num_dims, num_feat).repeat(1, 1, k, 1, 1)

    feature = torch.cat((feature - x, x), dim=3).permute(0, 3, 4, 1, 2).contiguous()

    return feature


def vn_get_graph_feature_no_x(x, k=20, idx=None, x_coord=None):
    batch_size, num_dims, num_feat, num_points = x.shape  # B,1,3,N

    x = x.view(batch_size, -1, num_points)
    if idx is None:
        if x_coord is None:  # dynamic knn graph
            idx = knn(x, k=k)
        else:  # fixed knn graph with input point coordinates
            idx = knn(x_coord, k=k)
    device = torch.device('cuda')
    try:
        idx_base = torch.arange(0, batch_size, device=device) * num_points
    except:
        print(batch_size)
    idx_base.unsqueeze_(1).unsqueeze_(2)

    idx = idx + idx_base

    idx = idx.view(-1).contiguous()

    #_, num_dims, _ = x.size()
    #num_dims = num_dims // 3

    x = x.transpose(2, 1).contiguous()
    feature = x.view(batch_size * num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims, num_feat)
    x = x.view(batch_size, num_points, 1, num_dims, num_feat).repeat(1, 1, k, 1, 1)

    feature = (feature - x)

    return feature.permute(0, 3, 4, 1, 2).contiguous()

def vn_get_graph_feature2(x, feat, k=20, idx=None, x_coord=None):
    batch_size, num_dims, num_feat, num_points = x.shape  # B,1,3,N

    x = x.view(batch_size, -1, num_points)

    if idx is None:
        if x_coord is None:  # dynamic knn graph
            idx = knn(x, k=k)
        else:  # fixed knn graph with input point coordinates
            idx = knn(x_coord, k=k)
    device = torch.device('cuda')

    idx_base = torch.arange(0, batch_size, device=device) * num_points
    idx_base.unsqueeze_(1).unsqueeze_(2)

    idx = idx + idx_base

    idx = idx.view(-1).contiguous()

    #_, num_dims, _ = x.size()
    #num_dims = num_dims // 3

    x = x.transpose(2, 1).contiguous()
    x_ = x.view(batch_size * num_points, -1)[idx, :]
    x_ = x_.view(batch_size, num_points, k, num_dims, num_feat).permute(0, 3, 4, 1, 2).contiguous()
    x = x.view(batch_size, num_points, 1, num_dims, num_feat).repeat(1, 1, k, 1, 1).permute(0, 3, 4, 1, 2).contiguous()

    feat_ = None
    if feat is not None:

        _, num_dims_feat, num_featfeat, _ = feat.shape
        feat = feat.view(batch_size, -1, num_points)

        feat = feat.transpose(2,1).contiguous()
        feat_ = feat.view(batch_size * num_points, -1)[idx, :]
        feat_ = feat_.view(batch_size, num_points, k, num_dims_feat, num_featfeat).permute(0, 3, 4, 1, 2).contiguous()

        feat = feat.view(batch_size, num_points, 1, num_dims_feat, num_featfeat).repeat(1, 1, k, 1, 1).permute(0, 3, 4, 1, 2).contiguous()

    return x, x_, feat, feat_



def hn_get_graph_feature2(x, feat=None, sfeat=None, k=20, idx=None, x_coord=None):
    batch_size, num_dims, num_xfeat, num_points = x.shape  # B,1,3,N

    x = x.view(batch_size, -1, num_points)

    if idx is None:
        if x_coord is None:  # dynamic knn graph
            idx = knn(x, k=k)
        else:  # fixed knn graph with input point coordinates
            idx = knn(x_coord, k=k)
    device = torch.device('cuda')

    idx_base = torch.arange(0, batch_size, device=device) * num_points
    idx_base.unsqueeze_(1).unsqueeze_(2)

    idx = idx + idx_base

    idx = idx.view(-1).contiguous()

    #_, num_dims, _ = x.size()
    #num_dims = num_dims // 3

    x = x.transpose(2, 1).contiguous()
    x_ = x.view(batch_size * num_points, -1)[idx, :]
    x_ = x_.view(batch_size, num_points, k, num_dims, num_xfeat).permute(0, 3, 4, 1, 2).contiguous()
    x = x.view(batch_size, num_points, 1, num_dims, num_xfeat).repeat(1, 1, k, 1, 1).permute(0, 3, 4, 1, 2).contiguous()

    feat_ = None
    sfeat_ = None
    if feat is not None:

        _, num_dims_feat, feat_dim, _ = feat.shape
        feat = feat.view(batch_size, -1, num_points)

        feat = feat.transpose(2,1).contiguous()
        feat_ = feat.view(batch_size * num_points, -1)[idx, :]
        feat_ = feat_.view(batch_size, num_points, k, num_dims_feat, feat_dim).permute(0, 3, 4, 1, 2).contiguous()

        feat = feat.view(batch_size, num_points, 1, num_dims_feat, feat_dim).repeat(1, 1, k, 1, 1).permute(0, 3, 4, 1, 2).contiguous()
    if sfeat is not None:
        _, num_dims_sfeat, _ = sfeat.shape # B, nfeat, num_points
        sfeat = sfeat.view(batch_size, -1, num_points)

        sfeat = sfeat.transpose(2,1).contiguous()
        sfeat_ = sfeat.view(batch_size * num_points, -1)[idx, :]
        sfeat_ = sfeat_.view(batch_size, num_points, k, num_dims_sfeat).permute(0, 3, 1, 2).contiguous()

        sfeat = sfeat.view(batch_size, num_points, 1, num_dims_sfeat).repeat(1, 1, k, 1).permute(0, 3, 1, 2).contiguous()
    return x, x_, feat, feat_, sfeat, sfeat_


def vn_get_graph_feature3(x, y, yfeat, k=20, idx=None, dim9=False):
    '''
    used when target has extra feature dims
    :param x: (B, 3,N1)
    :param y:(B, 3,N2)
    :param xfeat:(B, d,N1)
    :param yfeat:(B, d, N2)
    :param k:
    :param idx:
    :param dim9:
    :return:
    '''
    batch_size, num_dims, num_xfeat, num_points1 = x.shape # B,1,3,N

    _, num_dims_yfeat, num_yfeat, num_points2 = yfeat.shape

    x = x.view(batch_size, -1, num_points1)
    y = y.view(batch_size, -1, num_points2)
    yfeat = yfeat.view(batch_size, -1, num_points2)
    if idx is None:
        if dim9 == False:
            idx = knn2(x,y, k=k)  # (batch_size, num_points, k)
        else:
            idx = knn2(x[:, 6:], y[:, 6:], k=k)
    device = torch.device('cuda')

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points2

    idx = idx + idx_base

    idx = idx.view(-1)

    x = x.contiguous()  # (batch_size, num_dims, num_points)  -> (batch_size, n_points, n_dims)
    y = y.transpose(2,
                    1).contiguous()
    yfeat = yfeat.transpose(2,
                    1).contiguous()
    y = y.view(batch_size * num_points2, -1)[idx, :]
    y = y.view(batch_size, num_points1, k, num_dims,num_yfeat).permute(0,3,4,1,2)
    yfeature = yfeat.view(batch_size * num_points2, -1)[idx, :]
    yfeature = yfeature.view(batch_size, num_points1, k, num_dims_yfeat, 3).permute(0,3,4,1,2)

    x = x.view(batch_size, num_dims, num_xfeat, num_points1, 1).repeat(1, 1, 1, 1, k)

    edge = y - x
    #feature = torch.cat((feature - x, x), dim=3)
    #feature = torch.cat((feature, yfeature), dim=3).permute(0, 3, 1, 2).contiguous()

    return edge.contiguous(), x, y, yfeature.contiguous()  # [24, 2048, 20, 3] [24, 2048, 20, 3] [24, 2048, 20, 32]

def vn_get_graph_feature4(x, y, xfeat, yfeat, k=20, idx=None, dim9=False):
    '''
    used when target has extra feature dims
    :param x: (B, 3,N1)
    :param y:(B, 3,N2)
    :param xfeat:(B, d,N1)
    :param yfeat:(B, d, N2)
    :param k:
    :param idx:
    :param dim9:
    :return:
    '''
    batch_size, num_dims, _, num_points1 = x.shape # B,1,3,N
    _, num_dims_xfeat, num_xfeat, _ = xfeat.shape

    _, num_dims_yfeat, num_yfeat, num_points2 = yfeat.shape

    x = x.view(batch_size, -1, num_points1)
    y = y.view(batch_size, -1, num_points2)
    xfeat = xfeat.reshape(batch_size, -1, num_points1)
    yfeat = yfeat.reshape(batch_size, -1, num_points2) # ???
    if idx is None:
        if dim9 == False:
            idx = knn2(x,y, k=k)  # (batch_size, num_points, k)
        else:
            idx = knn2(x[:, 6:], y[:, 6:], k=k)
    device = torch.device('cuda')

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points2

    idx = idx + idx_base

    idx = idx.view(-1)

    x = x.contiguous()  # (batch_size, num_dims, num_points)  -> (batch_size, n_points, n_dims)
    xfeat = xfeat.contiguous()  # (batch_size, num_dims, num_points)  -> (batch_size, n_points, n_dims)
    y = y.transpose(2,1).contiguous()
    yfeat = yfeat.transpose(2,1).contiguous()
    y = y.view(batch_size * num_points2, -1)[idx, :]
    y = y.view(batch_size, num_points1, k, num_dims,num_yfeat).permute(0,3,4,1,2)
    yfeature = yfeat.view(batch_size * num_points2, -1)[idx, :]
    yfeature = yfeature.view(batch_size, num_points1, k, num_dims_yfeat, 3).permute(0,3,4,1,2)

    x = x.view(batch_size, num_dims, num_xfeat, num_points1, 1).repeat(1, 1, 1, 1, k)
    xfeat = xfeat.view(batch_size, num_dims_xfeat, num_xfeat, num_points1, 1).repeat(1, 1, 1, 1, k)

    #feature = torch.cat((feature - x, x), dim=3)
    #feature = torch.cat((feature, yfeature), dim=3).permute(0, 3, 1, 2).contiguous()

    return x, y, xfeat.contiguous(), yfeature.contiguous()  # [24, 2048, 20, 3] [24, 2048, 20, 3] [24, 2048, 20, 32]



def hn_get_graph_feature4(x, y, xfeat=None, yfeat=None, sxfeat=None, syfeat=None, k=20, idx=None, dim9=False):
    '''
    used when target has extra feature dims
    :param x: (B, 3,N1)
    :param y:(B, 3,N2)
    :param xfeat:(B, d,N1)
    :param yfeat:(B, d, N2)
    :param k:
    :param idx:
    :param dim9:
    :return:
    '''
    batch_size, num_dims, num_xfeat, num_points1 = x.shape # B,1,3,N
    num_points2 = y.shape[3]

    x = x.view(batch_size, -1, num_points1)
    y = y.view(batch_size, -1, num_points2)
    if idx is None:
        if dim9 == False:
            idx = knn2(x,y, k=k)  # (batch_size, num_points, k)
        else:
            idx = knn2(x[:, 6:], y[:, 6:], k=k)
    device = torch.device('cuda')
    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points2
    idx = idx + idx_base
    idx = idx.view(-1)

    x = x.contiguous()  # (batch_size, num_dims, num_points)  -> (batch_size, n_points, n_dims)
    x = x.view(batch_size, num_dims, num_xfeat, num_points1, 1).repeat(1, 1, 1, 1, k)

    y = y.transpose(2,1).contiguous()
    y = y.view(batch_size * num_points2, -1)[idx, :]
    y = y.view(batch_size, num_points1, k, num_dims,3).permute(0,3,4,1,2)

    if xfeat is not None:
        _, num_dims_xfeat, xfeat_featdim, _ = xfeat.shape
        xfeat = xfeat.reshape(batch_size, -1, num_points1)
        xfeat = xfeat.contiguous()  # (batch_size, num_dims, num_points)  -> (batch_size, n_points, n_dims)
        xfeat = xfeat.view(batch_size, num_dims_xfeat, xfeat_featdim, num_points1, 1).repeat(1, 1, 1, 1, k).contiguous()

    if yfeat is not None:
        _, num_dims_yfeat, yfeat_featdim, _ = yfeat.shape
        yfeat = yfeat.reshape(batch_size, -1, num_points2) # ???
        yfeat = yfeat.transpose(2,1).contiguous()
        yfeat = yfeat.view(batch_size * num_points2, -1)[idx, :]
        yfeat = yfeat.view(batch_size, num_points1, k, num_dims_yfeat, yfeat_featdim).permute(0,3,4,1,2).contiguous()
    if sxfeat is not None:
        _, num_dims_sxfeat, _ = sxfeat.shape # B, nfeat, num_points
        sxfeat = sxfeat.transpose(2,1).contiguous()
        sxfeat = sxfeat.view(batch_size, num_points1, 1, num_dims_sxfeat).repeat(1, 1, k, 1).permute(0, 3, 1, 2).contiguous()

    if syfeat is not None:
        _, num_dims_syfeat, _ = syfeat.shape # B, nfeat, num_points
        syfeat = syfeat.transpose(2,1).contiguous()
        syfeat = syfeat.view(batch_size * num_points2, -1)[idx, :]
        syfeat = syfeat.view(batch_size, num_points1, k, num_dims_syfeat).permute(0,3,1,2).contiguous()

    return x, y, xfeat, yfeat, sxfeat, syfeat  # [24, 2048, 20, 3] [24, 2048, 20, 3] [24, 2048, 20, 32]


class VNLinear(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(VNLinear, self).__init__()
        self.map_to_feat = nn.Linear(in_channels, out_channels, bias=False)

    def forward(self, x):
        '''
        x: point features of shape [B, N_feat, 3, N_samples, ...]
        '''
        x_out = self.map_to_feat(x.transpose(1, -1)).transpose(1, -1)
        return x_out



class VNLinearLeakyReLU(nn.Module):
    def __init__(self, in_channels, out_channels, rot_configs, args, dim=5, share_nonlinearity=False, negative_slope=0.2, bn=False, rot=False, use_batchnorm=False, v_nonlinearity=True):
        super().__init__()
        self.dim = dim
        self.negative_slope = negative_slope
        self.v_nonlinearity = v_nonlinearity
        self.args = args

        # self.fer = evl.MakeHDFeature(rot_configs, args, in_channels)
        self.map_to_feat = nn.Linear(in_channels, out_channels, bias=False)
        
        self.bn=bn

        if args.vn_nonlinearity == 'fer':
            self.fer_nonlinearity = evl.EVNNonLinearity(out_channels, args)
        elif args.vn_nonlinearity == 'vn':
            if share_nonlinearity == True:
                self.map_to_dir = nn.Linear(in_channels, 1, bias=False)
                self.rot = False
            else:
                self.map_to_dir = nn.Linear(in_channels, out_channels, bias=False)

    def forward(self, x):
        '''
        x: point features of shape [B, N_feat, 3, N_samples, ...]
        '''
        # if x.shape[2] == 3:
        #     x = self.fer(x)
        # Linear
        p = self.map_to_feat(x.transpose(1, -1)).transpose(1, -1)
        # BatchNorm
        if self.v_nonlinearity:
            if self.args.vn_nonlinearity == 'vn':
                # LeakyReLU
                d = self.map_to_dir(x.transpose(1, -1)).transpose(1, -1)
                dotprod = (p * d).sum(2, keepdims=True)
                mask = (dotprod < 0).float()
                d_norm_sq = torch.pow(torch.norm(d, 2, dim=2, keepdim=True),2)
                x_out = p - (mask) * (1-self.negative_slope) * (dotprod / (d_norm_sq + EPS2)) * d
                # del mask
            elif self.args.vn_nonlinearity == 'fer':
                x_out = self.fer_nonlinearity(p)
        else:
            x_out = p
        return x_out# todo 改过
    

# class VNLinearLeakyReLU(nn.Module):
#     def __init__(self, in_channels, out_channels, args, rot_configs, dim=5, share_nonlinearity=False, negative_slope=0.2, bn=False, rot=False, use_batchnorm=False, v_nonlinearity=True):
#         super().__init__()
#         self.dim = dim
#         self.negative_slope = negative_slope
#         self.v_nonlinearity = v_nonlinearity

#         # self.fer = evl.MakeHDFeature(rot_configs, args, in_channels)
#         self.map = evl.EVLinearLeakyReLU(in_channels, out_channels, args)

#     def forward(self, x):
#         '''
#         x: point features of shape [B, N_feat, 3, N_samples, ...]
#         '''
#         # if x.shape[2] == 3:
#         #     x = self.fer(x)
#         return self.map(x)


class HNLinearLeakyReLU(nn.Module):
    def __init__(self, v_in_channels, v_out_channels, s_in_channels=0, s_out_channels=0, args=None, rot_configs=None, dim=5, share_nonlinearity=False, negative_slope=0.2, \
                 bn=False, rot=False, is_s2v=True, bias=False, scale_equivariance=False, v2s_norm=True, p_norm=1, v_nonlinearity=True): # todo v2s_norm true 1102
        super().__init__()
        self.dim = dim
        self.negative_slope = negative_slope
        self.v_nonlinearity = v_nonlinearity
        self.args = args

        # self.fer = evl.MakeHDFeature(rot_configs, args, v_in_channels)
        self.fer_nonlinearity = evl.EVNNonLinearity(v_out_channels, args)

        self.map_to_feat = nn.Linear(v_in_channels, v_out_channels, bias=False)
        if self.v_nonlinearity:
            if share_nonlinearity == True:
                self.map_to_dir = nn.Linear(v_in_channels, 1, bias=False)
                self.rot = False
            else:
                self.map_to_dir = nn.Linear(v_in_channels, v_out_channels, bias=False)

        if s_in_channels > 0:
            # scalar
            if s_out_channels > 0:
                self.ss = nn.Linear(s_in_channels, s_out_channels, bias=True) # todo bias
            if is_s2v:
                self.sv = nn.Linear(s_in_channels, v_out_channels, bias=True) # todo bias
            if bn: # todo xuyaogai
                self.s_bn = nn.BatchNorm1d(s_out_channels)
        if s_out_channels > 0:
            # self.v2s = VNStdFeature(v_in_channels, args=args, rot_configs=rot_configs, dim=dim, ver=1, reduce_dim2=True, regularize=True, scale_equivariance=scale_equivariance)
            self.v2s = VNStdFeature(v_in_channels, args=args, rot_configs=rot_configs, dim=dim, ver=1, reduce_dim2=True, regularize=False, scale_equivariance=False)
            self.vs = nn.Linear(v_in_channels, s_out_channels, bias=True) # todo bias
            if s_in_channels == 0:
                self.sv = nn.Linear(s_out_channels, v_out_channels, bias=True) # todo bias

        self.s_in_channels = s_in_channels
        self.s_out_channels = s_out_channels
        self.v_in_channels = v_in_channels
        self.v_out_channels = v_out_channels

        self.is_s2v=is_s2v
        self.v2s_norm = v2s_norm # todo new 1102
        self.p_norm=p_norm # used in s2v

        self.scale_equivariance = scale_equivariance


    def forward(self, x, s=None):
        '''
        x: point features of shape [B, N_feat, 3, N_samples, ...]
        s: scalar point features of shape [B, N_feat, N_samples, ...]
        '''

        # Linear
        p = self.map_to_feat(x.transpose(1, -1)).transpose(1, -1)

        if s is not None:
            if self.is_s2v:
                sv = self.sv(s.transpose(1, -1)).transpose(1, -1).unsqueeze(2)
                if self.v2s_norm:
                    p = p * sv / (sv.norm(p=self.p_norm, dim=1, keepdim=True)/self.v_out_channels + EPS)  # todo if add regularization
                else:
                    p = p * sv / ( sv.norm(p=1, dim=1, keepdim=True) + EPS ) # todo if add regularization

            if self.s_out_channels > 0:
                ss = self.ss(s.transpose(1, -1)).transpose(1, -1)
                vs = self.vs(self.v2s(x)[0].transpose(1, -1)).transpose(1, -1)
                s = F.leaky_relu(ss+vs, self.negative_slope)

        elif self.s_out_channels > 0:
            vs = self.vs(self.v2s(x)[0].transpose(1, -1)).transpose(1, -1)
            s = F.leaky_relu(vs, self.negative_slope)
            sv = self.sv(s.transpose(1, -1)).transpose(1, -1)
            p = p * sv.unsqueeze(2)

        # if self.v_nonlinearity:
        if self.args.hn_nonlinearity == 'vn':
            # LeakyReLU
            d = self.map_to_dir(x.transpose(1, -1)).transpose(1, -1)
            d_norm_sq = torch.pow(torch.norm(d, 2, dim=2, keepdim=True), 2)
            dotprod = (p * d).sum(2, keepdims=True)
            mask = (dotprod < 0).float()
            x_out = p - (mask) * (1 - self.negative_slope) * (dotprod / (d_norm_sq + EPS2)) * d
            # del mask
        elif self.args.hn_nonlinearity == 'fer':
            x_out = self.fer_nonlinearity(p)
        else:
            x_out = p
        return x_out, s# todo 改过


def mean_pool(x, dim=-1, keepdim=False):
    return x.mean(dim=dim, keepdim=keepdim)


class VNStdFeature(nn.Module): # also for HNStdFeature
    '''
    ver==0: old
    ver==1: z dir :== mean at N_feat
    ver==2: length of
    '''
    def __init__(self, in_channels, args, rot_configs, dim=4, normalize_frame=False, share_nonlinearity=False, negative_slope=0.2, ver=0, scale_equivariance=False, reduce_dim2=False, regularize=True): # todo regularize changed 1102
        super().__init__()
        self.ver = ver
        self.dim = dim
        self.scale_equivariance=scale_equivariance
        self.reduce_dim2=reduce_dim2 # only work with ver1
        self.regularize = regularize
        self.in_channels = in_channels
    def forward(self, x, s=None):
        '''
        x: point features of shape [B, N_feat, 3, N_samples, ...]
        '''
        z0 = torch.mean(x, dim=1, keepdim=True)
        # if not self.reduce_dim2: # always
        #     shape = x[:, :3, ...].shape
        #     z0 = z0.expand(shape)
        z0 = z0.transpose(1, 2)

        if self.ver==0 or self.ver==1:
            if self.dim == 4:
                x_std = torch.einsum('bijm,bjkm->bikm', x, z0)
            elif self.dim == 3:
                x_std = torch.einsum('bij,bjk->bik', x, z0)
            elif self.dim == 5:
                x_std = torch.einsum('bijmn,bjkmn->bikmn', x, z0)

        # if self.regularize:
        #     x_std = x_std / (z0.transpose(1,2).norm(p=2, dim=2, keepdim=True) + EPS)

        if self.reduce_dim2:
            assert self.ver==1
            x_std = x_std.squeeze(2)
        if s is not None:
            assert self.reduce_dim2
            x_std = torch.cat((x_std, s), dim=1)
        return x_std, z0
    # ver1 reduce dim2 B,nfeat,1,xxx => B,nfeat,xxx
    # ver1 not reduce dim2 B,nfeat,3,xxx
    # ver0 B,nfeat,3,xxx




def get_graph_feature_cross(x, k=20, idx=None):
    batch_size = x.size(0)
    num_points = x.size(3)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        idx = knn(x, k=k)
    device = torch.device('cuda')

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points

    idx = idx + idx_base

    idx = idx.view(-1)

    _, num_dims, _ = x.size()
    num_dims = num_dims // 3

    x = x.transpose(2, 1).contiguous()
    feature = x.view(batch_size * num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims, 3)
    x = x.view(batch_size, num_points, 1, num_dims, 3).repeat(1, 1, k, 1, 1)
    cross = torch.cross(feature, x, dim=-1)

    feature = torch.cat((feature - x, x, cross), dim=3).permute(0, 3, 4, 1, 2).contiguous()

    return feature