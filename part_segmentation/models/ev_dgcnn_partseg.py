import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from .ev_layers import *
from .utils.ev_util import *

import models.utils.rotm_util as rmutil

class EV_DGCNN_PSEG(nn.Module):
    def __init__(self, args, seg_num_all):
        super(EV_DGCNN_PSEG, self).__init__()
        self.args = args
        self.seg_num_all = seg_num_all
        self.k = args.k

        stdim = int(args.rot_order.split('-')[0])
        edim = int(args.rot_order.split('-')[1])
        dim_list = list(range(stdim, edim+1))

        rot_configs = rmutil.init_rot_config(args.seed, dim_list, args.rot_type)
        rot_configs['constant_scale'] = 0
        self.rot_configs = rot_configs
        feat_dim = np.sum(2*np.array(rot_configs['dim_list'])+1)
        feat_dim = np.minimum(feat_dim, 20)
        self.feat_dim = feat_dim
        
        args.feat_dim = feat_dim
        args.dim_list = dim_list
        
        self.make_feature = MakeHDFeature(self.rot_configs, args, 3)

        self.bn7 = nn.BatchNorm1d(64)
        self.bn8 = nn.BatchNorm1d(256)
        self.bn9 = nn.BatchNorm1d(256)
        self.bn10 = nn.BatchNorm1d(128)
        
        self.conv1 = EVLinearLeakyReLU(3, 64//feat_dim, args)
        self.conv2 = EVLinearLeakyReLU(64//feat_dim, 64//feat_dim, args)
        self.conv3 = EVLinearLeakyReLU(64//feat_dim*2, 64//feat_dim, args)
        self.conv4 = EVLinearLeakyReLU(64//feat_dim, 64//feat_dim, args)
        self.conv5 = EVLinearLeakyReLU(64//feat_dim*2, 64//feat_dim, args)
        
        self.conv6 = EVLinearLeakyReLU(64//feat_dim*3, 1024//feat_dim, args)
        self.std_feature = EVNStdFeature(1024//feat_dim*2, args, dim=4)
        
        cat_dim = (64//feat_dim * 3) * feat_dim \
                + 1024//feat_dim*2 * feat_dim + 64

        self.conv8 = nn.Sequential(nn.Conv1d(cat_dim, 256, kernel_size=1, bias=False),
                               self.bn8,
                               nn.LeakyReLU(negative_slope=0.2))
        
        self.conv7 = nn.Sequential(nn.Conv1d(16, 64, kernel_size=1, bias=False),
                                   self.bn7,
                                   nn.LeakyReLU(negative_slope=0.2))
        
        self.dp1 = nn.Dropout(p=0.5)
        self.conv9 = nn.Sequential(nn.Conv1d(256, 256, kernel_size=1, bias=False),
                                   self.bn9,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.dp2 = nn.Dropout(p=0.5)
        self.conv10 = nn.Sequential(nn.Conv1d(256, 128, kernel_size=1, bias=False),
                                   self.bn10,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv11 = nn.Conv1d(128, self.seg_num_all, kernel_size=1, bias=False)
        

    def forward(self, x, l):
        batch_size = x.size(0)
        num_points = x.size(2)
        
        x = x.unsqueeze(1)
        x = get_graph_feature_cross(x, k=self.k)
        x = self.make_feature(x)

        x = self.conv1(x)
        x = self.conv2(x)
        x1 = torch.mean(x, -1)
        
        x = get_graph_feature(x1, k=self.k)
        x = self.conv3(x)
        x = self.conv4(x)
        x2 = torch.mean(x, -1)
        
        x = get_graph_feature(x2, k=self.k)
        x = self.conv5(x)
        x3 = torch.mean(x, -1)

        x123 = torch.cat((x1, x2, x3), dim=1)
        
        x = self.conv6(x123)
        x_mean = x.mean(dim=-1, keepdim=True).expand(x.size())
        x = torch.cat((x, x_mean), 1)
        x, z0 = self.std_feature(x)
        x123 = torch.einsum('bijm,bjkm->bikm', x123, z0).contiguous().view(batch_size, -1, num_points)
        x = x.view(batch_size, -1, num_points)
        x = x.max(dim=-1, keepdim=True)[0]

        l = l.view(batch_size, -1, 1)
        l = self.conv7(l)

        x = torch.cat((x, l), dim=1)
        x = x.repeat(1, 1, num_points)

        x = torch.cat((x, x123), dim=1)

        x = self.conv8(x)
        x = self.dp1(x)
        x = self.conv9(x)
        x = self.dp2(x)
        x = self.conv10(x)
        x = self.conv11(x)

        return x
    