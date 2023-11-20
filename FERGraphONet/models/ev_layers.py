"""
"""
import numpy as np

import torch
import torch.nn as nn
from .utils.ev_util import *

EPS = 1e-6

class MakeHDFeature(nn.Module):
    def __init__(self, rot_configs, args, nfeat=2, feature_axis=2):
        super(MakeHDFeature, self).__init__()
        self.rot_configs = rot_configs
        self.args = args
        self.radial_func = nn.ModuleList()
        self.feature_axis = feature_axis

        if self.args.psi_scale_type in [2,3]:
            for _ in self.rot_configs['dim_list']:
                mixing_dim = nfeat if self.args.psi_scale_type==3 else 1
                self.radial_func.append(nn.Sequential(
                    nn.Linear(mixing_dim,16),
                    nn.LeakyReLU(args.negative_slope),
                    nn.Linear(16,16),
                    nn.LeakyReLU(args.negative_slope),
                    nn.Linear(16,mixing_dim),
                ))
        elif self.args.psi_scale_type in [0,1]:
            self.scale_params = nn.Parameter(data=torch.Tensor(torch.ones(len(self.rot_configs['dim_list']))), requires_grad=True)
        
    def forward(self, x):
        x = torch.transpose(x, self.feature_axis, -1)
        x_norm = x.norm(dim=-1, keepdim=True) + EPS
        x_hat = x/x_norm
        ft_list = []
        for i, dl in enumerate(self.rot_configs['dim_list']):
            x_ = Y_func_V2(dl, x_hat, self.rot_configs)
            if self.args.psi_scale_type == 0:
                x_norm_ = x_norm * self.scale_params[i]
            elif self.args.psi_scale_type == 1:
                x_norm_ = x_norm**dl * self.scale_params[i]
            elif self.args.psi_scale_type == 2:
                x_norm_ = self.radial_func[i](x_norm)
            elif self.args.psi_scale_type == 3:
                x_norm_ = self.radial_func[i](x_norm.transpose(1,-1)).transpose(1,-1)
            x_ = x_*x_norm_
            ft_list.append(x_)

        feat = torch.concat(ft_list, -1)
        return feat.transpose(self.feature_axis, -1).contiguous()



class EVNLinear(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(EVNLinear, self).__init__()
        self.map_to_feat = nn.Linear(in_channels, out_channels, bias=False)
    
    def forward(self, x):
        x_out = self.map_to_feat(x.transpose(1,-1)).transpose(1,-1)
        return x_out

class EVNNonLinearity(nn.Module):
    def __init__(self, num_features, args):
        super(EVNNonLinearity, self).__init__()
        self.args = args
        
        self.gate = nn.Sequential(
            nn.Linear(num_features, num_features//2),
            nn.LeakyReLU(args.negative_slope),
            nn.Linear(num_features//2, num_features),
            )
    
    def forward(self, x):
        '''
        x: point features of shape [B, D, N, ...]
        '''
        norm = torch.norm(x, dim=2) + EPS # [B D N ...]
        if self.args.residual:
            norm_bn = self.gate(norm.transpose(1,-1)).transpose(1,-1)+norm
        else:
            norm_bn = self.gate(norm.transpose(1,-1)).transpose(1,-1)
        norm = norm.unsqueeze(2)
        norm_bn = norm_bn.unsqueeze(2)
        x = x / norm * norm_bn
        return x

        
class EVLinearLeakyReLU(nn.Module):
    def __init__(self, in_channels, out_channels, args):
        super(EVLinearLeakyReLU, self).__init__()
        self.negative_slope = args.negative_slope
        
        self.map_to_feat = nn.Linear(in_channels, out_channels, bias=False)
        self.nonlinearity = EVNNonLinearity(out_channels, args)
    
    def forward(self, x):
        '''
        x: point features of shape [B, N_feat, 3, N_samples, ...]
        '''
        # Linear
        p = self.map_to_feat(x.transpose(1,-1)).transpose(1,-1)
        p = self.nonlinearity(p)
        return p
    

class EVNStdFeature(nn.Module):
    def __init__(self, in_channels, args, dim=4, out_feat_dim=None, normalize_frame=False, share_nonlinearity=False):
        super(EVNStdFeature, self).__init__()

        self.dim = dim
        self.normalize_frame = normalize_frame
        self.feat_dim = args.feat_dim
        if out_feat_dim is None:
            out_feat_dim = args.feat_dim
        
        self.vn1 = EVLinearLeakyReLU(in_channels, in_channels//2, args)
        self.vn2 = EVLinearLeakyReLU(in_channels//2, in_channels//4, args)
        self.vn_lin = nn.Linear(in_channels//4, out_feat_dim, bias=False)
    
    def forward(self, x):
        '''
        x: point features of shape [B, N_feat, 3, N_samples, ...]
        '''
        z0 = x
        z0 = self.vn1(z0)
        z0 = self.vn2(z0)
        z0 = self.vn_lin(z0.transpose(1, -1)).transpose(1, -1)
        z0 = z0.transpose(1, 2)
        
        if self.dim == 4:
            x_std = torch.einsum('bijm,bjkm->bikm', x, z0)
        elif self.dim == 3:
            x_std = torch.einsum('bij,bjk->bik', x, z0)
        elif self.dim == 5:
            x_std = torch.einsum('bijmn,bjkmn->bikmn', x, z0)
        
        return x_std.contiguous(), z0.contiguous()
    
def get_feat(x, l, rot_configs, feature_axis=-2):
    idx = rot_configs['dim_list'].index(l)
    sidx = np.sum(np.array(rot_configs['dim_list'][:idx]) * 2 + 1).astype(int)
    eidx = sidx + 2*l+1
    return x.transpose(feature_axis, -1)[..., sidx:eidx].transpose(feature_axis, -1)