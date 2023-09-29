import torch
import torch.nn as nn
from im2mesh.layers_vnn import *


def maxpool(x, dim=-1, keepdim=False):
    out, _ = x.max(dim=dim, keepdim=keepdim)
    return out


def meanpool(x, dim=-1, keepdim=False):
    out = x.mean(dim=dim, keepdim=keepdim)
    return out


class VNN_ResnetPointnet(nn.Module):
    ''' PointNet-based VNN encoder network with ResNet blocks.

    Args:
        c_dim (int): dimension of latent code c
        dim (int): input points dimension
        hidden_dim (int): hidden dimension of the network
    '''

    def __init__(self, c_dim=128, dim=3, hidden_dim=128, k=20, meta_output=None):
        super().__init__()
        self.c_dim = c_dim
        self.k = k
        self.meta_output = meta_output

        self.conv_pos = VNLinearLeakyReLU(3, 128, negative_slope=0.0, share_nonlinearity=False, use_batchnorm=False)
        self.fc_pos = VNLinear(128, 2*hidden_dim)
        self.block_0 = VNResnetBlockFC(2*hidden_dim, hidden_dim)
        self.block_1 = VNResnetBlockFC(2*hidden_dim, hidden_dim)
        self.block_2 = VNResnetBlockFC(2*hidden_dim, hidden_dim)
        self.block_3 = VNResnetBlockFC(2*hidden_dim, hidden_dim)
        self.block_4 = VNResnetBlockFC(2*hidden_dim, hidden_dim)
        self.fc_c = VNLinear(hidden_dim, c_dim)

        self.actvn_c = VNLeakyReLU(hidden_dim, negative_slope=0.0, share_nonlinearity=False)
        self.pool = meanpool
        
        if meta_output == 'invariant_latent':
            self.std_feature = VNStdFeature(c_dim, dim=3, normalize_frame=True, use_batchnorm=False)
        elif meta_output == 'invariant_latent_linear':
            self.std_feature = VNStdFeature(c_dim, dim=3, normalize_frame=True, use_batchnorm=False)
            self.vn_inv = VNLinear(c_dim, 3)
        elif meta_output == 'equivariant_latent_linear':
            self.vn_inv = VNLinear(c_dim, 3)

    def forward(self, p):
        batch_size = p.size(0)
        p = p.unsqueeze(1).transpose(2, 3)
        #mean = get_graph_mean(p, k=self.k)
        #mean = p_trans.mean(dim=-1, keepdim=True).expand(p_trans.size())
        feat = get_graph_feature_cross(p, k=self.k) # [B, 3, 3, N, K]
        net = self.conv_pos(feat)                   # [B, C, 3, N, K]
        net = self.pool(net, dim=-1)                # [B, C, 3, N]
        
        net = self.fc_pos(net)
        
        net = self.block_0(net)
        pooled = self.pool(net, dim=-1, keepdim=True).expand(net.size())
        net = torch.cat([net, pooled], dim=1)
        
        net = self.block_1(net)
        pooled = self.pool(net, dim=-1, keepdim=True).expand(net.size())
        net = torch.cat([net, pooled], dim=1)
        
        net = self.block_2(net)
        pooled = self.pool(net, dim=-1, keepdim=True).expand(net.size())
        net = torch.cat([net, pooled], dim=1)
        
        net = self.block_3(net)
        pooled = self.pool(net, dim=-1, keepdim=True).expand(net.size())
        net = torch.cat([net, pooled], dim=1)

        net = self.block_4(net)

        # Recude to  B x F
        net = self.pool(net, dim=-1)

        c = self.fc_c(self.actvn_c(net))
        
        if self.meta_output == 'invariant_latent':
            c_std, z0 = self.std_feature(c)
            return c, c_std
        elif self.meta_output == 'invariant_latent_linear':
            c_std, z0 = self.std_feature(c)
            c_std = self.vn_inv(c_std)
            return c, c_std
        elif self.meta_output == 'equivariant_latent_linear':
            c_std = self.vn_inv(c)
            return c, c_std

        return c
