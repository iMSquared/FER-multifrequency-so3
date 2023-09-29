import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict
from im2mesh.layers import (
    ResnetBlockFC
)
from im2mesh.layers_evn import EVNLinear, make_features, MakeHDFeature


class EVN_DecoderInner(nn.Module):
    ''' Decoder class.

    It does not perform any form of normalization.

    Args:
        dim (int): input dimension
        z_dim (int): dimension of latent code z
        c_dim (int): dimension of latent conditioned code c
        hidden_size (int): hidden size of Decoder network
        leaky (bool): whether to use leaky ReLUs
    '''

    def __init__(self, dim: int = 3, 
                       z_dim: int = 128, 
                       c_dim: int = 128,
                       hidden_size: int = 128, 
                       leaky: bool = False, 
                       use_scale_params: bool = False,
                       use_radial: bool = False,
                       rot_configs: Dict = None):
        super().__init__()
        self.z_dim = z_dim
        self.c_dim = c_dim        
        # Debug
        assert rot_configs is not None
        # EVN
        self.use_scale_params = use_scale_params
        self.use_radial       = use_radial
        self.rot_configs      = rot_configs

        # EVN: New feature
        if use_radial:
            self.make_features = MakeHDFeature(rot_configs, 1, 2)

        # EVN: Adjust scales at each feature spaces.
        if self.use_scale_params:
            num_rot_spaces = len(self.rot_configs["dim_list"])
            self.scale_params = nn.Parameter(
                data          = torch.Tensor(torch.ones(num_rot_spaces)),
                requires_grad = True)


        # Submodules
        if z_dim > 0:
            self.z_in = EVNLinear(z_dim, z_dim, rot_configs)
        if c_dim > 0:
            self.c_in = EVNLinear(c_dim, c_dim, rot_configs)
        
        self.fc_in = nn.Linear(z_dim*2+c_dim*2+1, hidden_size)

        self.block0 = ResnetBlockFC(hidden_size)
        self.block1 = ResnetBlockFC(hidden_size)
        self.block2 = ResnetBlockFC(hidden_size)
        self.block3 = ResnetBlockFC(hidden_size)
        self.block4 = ResnetBlockFC(hidden_size)

        self.fc_out = nn.Linear(hidden_size, 1)

        if not leaky:
            self.actvn = F.relu
        else:
            self.actvn = lambda x: F.leaky_relu(x, 0.2)

    def forward(self, p, z, c=None, **kwargs):
        batch_size, T, D = p.size()
        _, _, R = c.size()

        # Input 1) L2 Norm
        net = (p * p).sum(2, keepdim=True)

        # # We are not interested in generative model...
        # if self.z_dim != 0:
        #     z = z.view(batch_size, -1, D).contiguous()
        #     net_z = torch.einsum('bmi,bni->bmn', p, z)
        #     z_dir = self.z_in(z)
        #     z_inv = (z * z_dir).sum(-1).unsqueeze(1).repeat(1, T, 1)
        #     net = torch.cat([net, net_z, z_inv], dim=2)

        if self.c_dim != 0:
            # Input 2) Inner product at higher dimension
            c = c.contiguous()

            if self.use_radial:
                p_ext = p.transpose(-1,-2).unsqueeze(1)                         # [B, 1, 3, 2048]
                p_ext = self.make_features(p_ext).squeeze(1).transpose(-1, -2)  # [B, 3, 2048]
            else:
                if self.use_scale_params:
                    p_ext = make_features(p.transpose(-1, -2), 
                                            self.rot_configs, 
                                            self.scale_params).transpose(-1, -2)
                else:
                    p_ext = make_features(p.transpose(-1, -2), 
                                            self.rot_configs, 
                                            None).transpose(-1, -2)
                                
            net_c = torch.einsum('bmi,bni->bmn', p_ext, c)
            # Input 3) Invariant feature (compact version)
            c_dir = self.c_in(c)                        # [B, F, R]   
            c_inv = (c * c_dir).sum(-1)                 # [B, F]
            c_inv = c_inv.unsqueeze(1).repeat(1, T, 1)  # [B, T, F] Broadcast
            # Aggregate
            net = torch.cat([net, net_c, c_inv], dim=2)
        
        net = self.fc_in(net)

        net = self.block0(net)
        net = self.block1(net)
        net = self.block2(net)
        net = self.block3(net)
        net = self.block4(net)

        out = self.fc_out(self.actvn(net))
        out = out.squeeze(-1)

        return out