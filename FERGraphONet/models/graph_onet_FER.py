import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import einsum
from einops import rearrange, repeat
from pointnet2_ops import pointnet2_utils

import os, sys
BASEDIR = os.path.dirname(os.path.dirname(__file__))
if BASEDIR not in sys.path:
    sys.path.insert(0, BASEDIR)

import models.ev_layers as evl
import models.hn_fer_layers as hnl
from models.utils.ev_util import *
import models.utils.rotm_util as rmutil


def index_points(points, idx):
    """
    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points


class hn_DGCNN_spatial_fps_hier(nn.Module):
    def __init__(self, args, rot_configs, k=20, c_dim=128, bias=False, meanpool=True, selfconv=True, v_relu=True):

        super().__init__()
        self.c_dim = c_dim
        self.n_knn = k
        self.selfconv=selfconv
        self.base_dim = args.base_dim

        self.coord_map = evl.MakeHDFeature(rot_configs, args, 1)

        self.conv1 = hnl.HNLinearLeakyReLU(2, self.base_dim, s_in_channels=0, s_out_channels=32, args=args, rot_configs=rot_configs, bias=bias, v_nonlinearity=v_relu)
        self.init_coord_map = evl.MakeHDFeature(rot_configs, args, 2)
        self.conv2 = hnl.HNLinearLeakyReLU(self.base_dim, self.base_dim, s_in_channels=32, s_out_channels=32,args=args, rot_configs=rot_configs, bias=bias, v_nonlinearity=v_relu)
        self.conv3 = hnl.HNLinearLeakyReLU(self.base_dim * 2 + 2, self.base_dim, 32*2, 32, args=args, rot_configs=rot_configs, bias=bias, v_nonlinearity=v_relu)
        self.conv4 = hnl.HNLinearLeakyReLU(self.base_dim, self.base_dim, 32, 32, args=args, rot_configs=rot_configs, bias=bias, v_nonlinearity=v_relu)
        self.conv5 = hnl.HNLinearLeakyReLU(self.base_dim * 2 + 2, self.base_dim, 32*2, 32, args=args, rot_configs=rot_configs, bias=bias, v_nonlinearity=v_relu)

        if selfconv:
            self.selfconv1 = hnl.HNLinearLeakyReLU(1, self.base_dim, s_in_channels=0, s_out_channels=32, args=args, rot_configs=rot_configs, bias=bias, dim=4, share_nonlinearity=False, v_nonlinearity=v_relu)
            self.selfconv2 = hnl.HNLinearLeakyReLU(self.base_dim, self.base_dim, s_in_channels=32, s_out_channels=32, args=args, rot_configs=rot_configs, bias=bias, dim=4, share_nonlinearity=False, v_nonlinearity=v_relu)
            self.selfconv3 = hnl.HNLinearLeakyReLU(self.base_dim+1, self.base_dim, 32, 32, args=args, rot_configs=rot_configs, bias=bias, dim=4, share_nonlinearity=False, v_nonlinearity=v_relu)
            self.selfconv4 = hnl.HNLinearLeakyReLU(self.base_dim, self.base_dim, 32, 32, args=args, rot_configs=rot_configs, bias=bias, dim=4, share_nonlinearity=False, v_nonlinearity=v_relu)
            self.selfconv5 = hnl.HNLinearLeakyReLU(self.base_dim+1, self.base_dim, 32, 32, args=args, rot_configs=rot_configs, bias=bias, dim=4, share_nonlinearity=False, v_nonlinearity=v_relu)
        self.pool1 = hnl.mean_pool
        self.pool2 = hnl.mean_pool
        self.pool3 = hnl.mean_pool
        self.pool = hnl.mean_pool

        self.conv6 = hnl.HNLinearLeakyReLU(self.base_dim * 2, self.base_dim, 32*2, 32, dim=4, args=args, rot_configs=rot_configs, share_nonlinearity=False, bias=bias, v_nonlinearity=v_relu)
        self.conv7 = hnl.HNLinearLeakyReLU(self.base_dim * 2, self.base_dim, 32*2, 32, dim=4, args=args, rot_configs=rot_configs, share_nonlinearity=False, bias=bias, v_nonlinearity=v_relu)
        self.conv8 = hnl.HNLinearLeakyReLU(self.base_dim * 2, self.base_dim, 32*2, 32, dim=4, args=args, rot_configs=rot_configs, share_nonlinearity=False, bias=bias, v_nonlinearity=v_relu)
        
    def forward(self, x):
        #x  (B, N, 3)
        B, N1, _ = x.shape
        x = x.permute(0,2,1) #

        batch_size = x.size(0)
        num_points = x.size(2)

        x = x.unsqueeze(1) # B, 1, 3, N
        xyz1 = x.clone()
        xbak = x.clone()
        x = hnl.vn_get_graph_feature(x, k=self.n_knn)
        x = self.init_coord_map(x)
        x, s = self.conv1(x, None)
        x, s = self.conv2(x, s)
        x1 = self.pool1(x)
        s1 = s.max(dim=-1)[0]

        if self.selfconv:
            selfx, selfs = self.selfconv1(self.coord_map(xbak), None)
            selfx, selfs = self.selfconv2(selfx, selfs)
            x1 = (x1 + selfx).contiguous()
            s1 = (s1 + selfs).contiguous()

        N2 = int(N1/4)
        fps_ind1 = pointnet2_utils.furthest_point_sample(xyz1.permute(0,3,1,2).reshape(B,N1,3), N2)
        xyz2 = index_points(xyz1.permute(0,3,1,2).reshape(B,N1,3), fps_ind1).permute(0,2,1).unsqueeze(1)
        x = index_points(x1.permute(0,3,1,2), fps_ind1).permute(0,2,3,1)
        s = index_points(s1.permute(0,2,1), fps_ind1).permute(0,2,1)

        xbak = x.clone()
        sbak = s.clone()
        x_coord, y_coord, xfeat, yfeat, sxfeat, syfeat = hnl.hn_get_graph_feature2(xyz2, x, s, k=self.n_knn)
        x = torch.cat([self.coord_map(x_coord), self.coord_map(y_coord - x_coord), xfeat, yfeat],dim=1)
        s = torch.cat((sxfeat, syfeat), 1)
        x, s = self.conv3(x, s)
        x, s = self.conv4(x, s)
        x2 = self.pool2(x)
        s2 = s.max(dim=-1)[0]

        if self.selfconv:
            selfx, selfs = self.selfconv3(torch.cat((xbak, self.coord_map(xyz2)), dim=1), sbak)
            selfx, selfs = self.selfconv4(selfx, selfs)
            x2 = (x2 + selfx).contiguous()
            s2 = (s2 + selfs).contiguous()

        N3 = int(N2/4)
        fps_ind2 = pointnet2_utils.furthest_point_sample(xyz2.permute(0,3,1,2).reshape(B,N2,3), N3)
        xyz3 = index_points(xyz2.permute(0,3,1,2).reshape(B,N2,3), fps_ind2).permute(0,2,1).unsqueeze(1)
        x = index_points(x2.permute(0,3,1,2), fps_ind2).permute(0,2,3,1)
        s = index_points(s2.permute(0,2,1), fps_ind2).permute(0,2,1)

        xbak = x.clone()
        sbak = s.clone()

        x_coord, y_coord, xfeat, yfeat, sxfeat, syfeat = hnl.hn_get_graph_feature2(xyz3, x, s, k=self.n_knn)
        x = torch.cat([self.coord_map(x_coord), self.coord_map(y_coord - x_coord), xfeat, yfeat],dim=1)
        s = torch.cat((sxfeat, syfeat), 1)
        x, s = self.conv5(x, s)
        x3 = self.pool3(x)
        s3 = s.max(dim=-1)[0]

        if self.selfconv:
            selfx, selfs = self.selfconv5(torch.cat((xbak, self.coord_map(xyz3)), dim=1), sbak)
            x3 = (x3 + selfx).contiguous()
            s3 = (s3 + selfs).contiguous()

        x_max = self.pool(x3).unsqueeze(3).repeat(1,1,1,N3)
        s_max = s3.max(dim=-1)[0].unsqueeze(2).repeat(1,1,N3)

        x = torch.cat((x3, x_max),1)
        s = torch.cat((s3, s_max),1)
        x, s = self.conv6(x, s)

        _, _, xfeat, yfeat, sxfeat, syfeat = hnl.hn_get_graph_feature4(xyz2, xyz3, x2, x, s2, s, k=1)
        x = torch.cat((xfeat, yfeat), 1).squeeze_(4)
        s = torch.cat((sxfeat, syfeat), 1).squeeze_(3)
        x, s = self.conv7(x, s)

        _, _, xfeat, yfeat, sxfeat, syfeat = hnl.hn_get_graph_feature4(xyz1, xyz2, x1, x, s1, s, k=1)
        x = torch.cat((xfeat, yfeat), 1).squeeze_(4)
        s = torch.cat((sxfeat, syfeat), 1).squeeze_(3)
        x, s = self.conv8(x, s)

        return xyz1, x.contiguous(), s.contiguous(),  xyz2, x2.contiguous(), s2.contiguous(), xyz3, x3.contiguous(), s3.contiguous() # B,1,3,N, (B,1, 64//3, N)



# Resnet Blocks
class ResnetBlockFC(nn.Module):
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
        self.fc_0 = nn.Linear(size_in, size_h)
        self.fc_1 = nn.Linear(size_h, size_out)
        self.actvn = nn.ReLU()

        if size_in == size_out:
            self.shortcut = None
        else:
            self.shortcut = nn.Linear(size_in, size_out, bias=False)
        # Initialization
        nn.init.zeros_(self.fc_1.weight)

    def forward(self, x):
        net = self.fc_0(self.actvn(x))
        dx = self.fc_1(self.actvn(net))

        if self.shortcut is not None:
            x_s = self.shortcut(x)
        else:
            x_s = x

        return x_s + dx


class hn_DGCNN_decoder_hier2(nn.Module):
    ''' Decoder for PointConv Baseline.

    Args:
        dim (int): input dimension
        c_dim (int): dimension of latent conditioned code c
        hidden_size (int): hidden size of Decoder network
        leaky (bool): whether to use leaky ReLUs
        n_blocks (int): number of blocks ResNetBlockFC layers
        sample_mode (str): sampling mode  for points
    '''

    def __init__(self, args, rot_configs, k=20, c_dim=64, n_blocks=4, bias=False, meanpool=True, v_relu=True,**kwargs):
        super().__init__()
        self.c_dim = c_dim
        self.n_blocks = n_blocks
        self.k = k
        self.base_dim = args.base_dim

        self.coord_map = evl.MakeHDFeature(rot_configs, args, 2)

        self.conv1_1 = hnl.VNLinearLeakyReLU(2, 4, args=args, rot_configs=rot_configs, v_nonlinearity=v_relu)
        self.conv1_2 = hnl.HNLinearLeakyReLU(self.base_dim + 4, 4, 32, 32, args=args, rot_configs=rot_configs, bias=bias, v_nonlinearity=v_relu)
        self.conv1_3 = hnl.HNLinearLeakyReLU(4, 4, 32, 32, args=args, rot_configs=rot_configs, bias=bias, v_nonlinearity=v_relu)
        #self.pool1 = VNMaxPool(4)

        self.conv2_1 = hnl.VNLinearLeakyReLU(2, 4, args=args, rot_configs=rot_configs, v_nonlinearity=v_relu)
        self.conv2_2 = hnl.HNLinearLeakyReLU(self.base_dim + 4*2, 4, 32*2, 32, args=args, rot_configs=rot_configs, bias=bias, v_nonlinearity=v_relu)
        self.conv2_3 = hnl.HNLinearLeakyReLU(4, 4, 32, 32, args=args, rot_configs=rot_configs, bias=bias, v_nonlinearity=v_relu)
        #self.pool2 = VNMaxPool(4)

        self.conv3_1 = hnl.VNLinearLeakyReLU(2, 4, args=args, rot_configs=rot_configs, v_nonlinearity=v_relu)
        self.conv3_2 = hnl.HNLinearLeakyReLU(self.base_dim + 4*2, 4, 32*2, 32, args=args, rot_configs=rot_configs, bias=bias, v_nonlinearity=v_relu)
        self.conv3_3 = hnl.HNLinearLeakyReLU(4, 4, 32, 32, args=args, rot_configs=rot_configs, bias=bias, v_nonlinearity=v_relu)
        #self.pool3 = VNMaxPool(4)

        self.conv4_1 = hnl.VNLinearLeakyReLU(2, 4, args=args, rot_configs=rot_configs, v_nonlinearity=v_relu)
        self.conv4_2 = hnl.HNLinearLeakyReLU(self.base_dim + 4*2, 4, 32*2, 32, args=args, rot_configs=rot_configs, bias=bias, v_nonlinearity=v_relu)
        self.conv4_3 = hnl.HNLinearLeakyReLU(4, 4, 32, 32, args=args, rot_configs=rot_configs, bias=bias, v_nonlinearity=v_relu)
        #self.pool4 = VNMaxPool(4)

        self.pool1 = hnl.mean_pool
        self.pool2 = hnl.mean_pool
        self.pool3 = hnl.mean_pool
        self.pool4 = hnl.mean_pool

        self.conv1x1 = hnl.HNLinearLeakyReLU(4*3, self.base_dim, 32*3, 32, dim=4, args=args, rot_configs=rot_configs, share_nonlinearity=False, bias=bias, v_nonlinearity=v_relu)
        self.std_feature1 = hnl.VNStdFeature(self.base_dim, args=args, rot_configs=rot_configs, dim=4, normalize_frame=False, ver=1, reduce_dim2=True, regularize=True)

        if c_dim != 0:
            self.fc_c = nn.ModuleList([
                nn.Linear(32+self.base_dim, 32) for _ in range(n_blocks)
            ])

        self.fc_p1_1 = hnl.HNLinearLeakyReLU(1,self.base_dim,0,24, args=args, rot_configs=rot_configs, dim=4, share_nonlinearity=False, v_nonlinearity=v_relu)
        self.fc_p1_2 = hnl.HNLinearLeakyReLU(self.base_dim,self.base_dim,24,24, args=args, rot_configs=rot_configs, dim=4, share_nonlinearity=False, v_nonlinearity=v_relu)

        self.std_feature2 = hnl.VNStdFeature(self.base_dim, args=args, rot_configs=rot_configs, dim=4, normalize_frame=False, ver=1, reduce_dim2=True, regularize=True)
        # self.fc_p2 = nn.Linear(32, 32)
        self.fc_p2 = nn.Linear(self.base_dim+24, 32)

        self.blocks = nn.ModuleList([
            ResnetBlockFC(32),
            ResnetBlockFC(32),
            ResnetBlockFC(32),
            ResnetBlockFC(32),
        ])
        self.fc_out = nn.Linear(32, 1)

        self.actvn = lambda x: F.leaky_relu(x, 0.2)

    def forward(self, p, c, **kwargs):
        n_points = p.shape[1]  # =2048 p [B, N, 3], c[0] [B, N, 3], c[1] [B, N, 32]
        pp, fea, sfea, pp2, fea2, sfea2, pp3, fea3, sfea3 = c

        p = p.permute(0, 2, 1)
        p = p.unsqueeze(1).float()

        x, y, _, yfeature1, _, syfeature1 = hnl.hn_get_graph_feature4(p, pp, None, fea, None, sfea, k=self.k)  # (batch_size, 3, num_points) -> (batch_size, 3*2, num_points, k)

        x1 = torch.cat([self.conv1_1(self.coord_map(torch.cat([y-x, x], dim=1))),
                        yfeature1], dim=1)
        x1, s1 = self.conv1_2(x1, syfeature1)
        x1, s1 = self.conv1_3(x1, s1)
        x1 = self.pool1(x1)
        s1 = s1.max(dim=-1)[0]

        x2, y2, xfeature2, yfeature2, sxfeature2, syfeature2 = hnl.hn_get_graph_feature4(p, pp2, x1, fea2, s1,
                                                                                        sfea2, k=self.k)
        x2 = torch.cat([self.conv2_1(self.coord_map(torch.cat([y2-x2, x2], dim=1))), yfeature2, xfeature2], dim=1)
        x2, s2 = self.conv2_2(x2, torch.cat([syfeature2, sxfeature2], dim=1))
        x2, s2 = self.conv2_3(x2, s2)
        x2 = self.pool1(x2)
        s2 = s2.max(dim=-1)[0]

        x3, y3, xfeature3, yfeature3, sxfeature3, syfeature3 = hnl.hn_get_graph_feature4(p, pp3, x2, fea3, s2,
                                                                                        sfea3, k=self.k)
        x3 = torch.cat([self.conv3_1(self.coord_map(torch.cat([y3-x3, x3], dim=1))), yfeature3, xfeature3], dim=1)
        x3, s3 = self.conv3_2(x3, torch.cat([syfeature3, sxfeature3], dim=1))
        x3, s3 = self.conv3_3(x3, s3)
        x3 = self.pool1(x3)
        s3 = s3.max(dim=-1)[0]

        x3, y3, xfeature3, yfeature3, sxfeature3, syfeature3 = hnl.hn_get_graph_feature4(p, pp3, x3, fea3, s3,
                                                                                        sfea3, k=self.k)
        x3 = torch.cat([self.conv4_1(self.coord_map(torch.cat([y3-x3, x3], dim=1))), yfeature3, xfeature3], dim=1)
        x3, s3 = self.conv4_2(x3, torch.cat([syfeature3, sxfeature3], dim=1))
        x3, s3 = self.conv4_3(x3, s3)
        x3 = self.pool1(x3)
        s3 = s3.max(dim=-1)[0]
        c, s = self.conv1x1(torch.cat((x1, x2, x3), 1), torch.cat((s1, s2, s3), 1))
        c, _ = self.std_feature1(c, s=s)

        net, nets = self.fc_p1_1(p, None)
        net, nets = self.fc_p1_2(net, nets)
        net, _ = self.std_feature2(net, s=nets)
        #B, _, N = net.shape
        net = net.permute(0,2,1)
        net = self.fc_p2(net)
        # print(c.shape)
        c = c.permute(0,2,1)


        for i in range(self.n_blocks):
            if self.c_dim != 0:
                net = net + self.fc_c[i](c)

            net = self.blocks[i](net)

        out = self.fc_out(self.actvn(net))
        out = out.squeeze(-1)

        return out  # [B,2048]




if __name__ == '__main__':
    nb = 2
    npnt = 512
    input_pnts = torch.normal(0,1,size=(nb, npnt, 3)).to('cuda')
    qpnts = torch.normal(0,1,size=(nb, npnt, 3)).to('cuda')

    class Args:
        pass
    args = Args()
    args.rot_order = '1-2'
    args.seed = 1
    args.psi_scale_type = 3
    args.negative_slope = 0

    rot_configs = rmutil.init_rot_config(seed=args.seed, dim_list=args.rot_order, rot_type='custom')

    # GraphONetEncoder(npoints=npnt)
    enc = hn_DGCNN_spatial_fps_hier(args, rot_configs, c_dim=64).to('cuda')
    dec = hn_DGCNN_decoder_hier2(args, rot_configs, c_dim=64).to('cuda')

    encoded = enc(input_pnts)
    qres = dec(qpnts, encoded)

    print(1)
