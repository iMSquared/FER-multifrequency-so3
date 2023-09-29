import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from .ev_layers import *
from .utils.ev_util import *

import models.utils.rotm_util as rmutil


class EV_DGCNN_CLS(nn.Module):
    def __init__(self, args, num_class=40):
        super(EV_DGCNN_CLS, self).__init__()
        self.args = args
        self.k = args.k
        dim_list = list(range(int(args.rot_order.split('-')[0]), int(args.rot_order.split('-')[1])+1))
        args.dim_list = dim_list
        self.rot_configs = rmutil.init_rot_config(args.seed, dim_list, args.rot_type)
        self.rot_configs['constant_scale'] = args.psi_scale_type
        feat_dim = np.sum(2*np.array(dim_list)+1)
        args.feat_dim = feat_dim

        if args.psi_scale_type >= 2:
            self.make_feature = MakeHDFeature(self.rot_configs, args, 3)
        else:
            self.scale_params = nn.Parameter(data=torch.Tensor(torch.ones(len(self.rot_configs['dim_list']))), requires_grad=True)
            self.make_feature = lambda x: make_features(x, self.rot_configs, self.scale_params, 2)


        midfeat_dim = np.minimum(feat_dim, 8)
        self.conv1 = EVLinearLeakyReLU(3, 64//midfeat_dim, args)
        self.conv2 = EVLinearLeakyReLU(64//midfeat_dim*2, 64//midfeat_dim, args)
        self.conv3 = EVLinearLeakyReLU(64//midfeat_dim*2, 128//midfeat_dim, args)
        self.conv4 = EVLinearLeakyReLU(128//midfeat_dim*2, 256//midfeat_dim, args)

        self.conv5 = EVLinearLeakyReLU(256//midfeat_dim+128//midfeat_dim+64//midfeat_dim*2, 1024//feat_dim, args)

        self.std_feature = EVNStdFeature(1024//feat_dim*2, args, dim=4, normalize_frame=False)
        self.linear1 = nn.Linear((1024//feat_dim)*4*feat_dim, 512)
        
        self.bn1 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(p=0.5)
        self.linear2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.dp2 = nn.Dropout(p=0.5)
        self.linear3 = nn.Linear(256, num_class)
        
    def forward(self, x):


        batch_size = x.size(0)
        x = x.unsqueeze(1)
        x = get_graph_feature_cross(x, k=self.k)
        x = self.make_feature(x)
        x = self.conv1(x)
        x1 = torch.mean(x, -1)

        x = get_graph_feature(x1, k=self.k)
        x = self.conv2(x)
        x2 = torch.mean(x, -1)
        
        x = get_graph_feature(x2, k=self.k)
        x = self.conv3(x)
        x3 = torch.mean(x, -1)
        
        x = get_graph_feature(x3, k=self.k)
        x = self.conv4(x)
        x4 = torch.mean(x, -1)
        x = torch.cat((x1, x2, x3, x4), dim=1)

        x = self.conv5(x)
        
        num_points = x.size(-1)
        x_mean = x.mean(dim=-1, keepdim=True).expand(x.size())
        x = torch.cat((x, x_mean), 1)
        x, trans = self.std_feature(x)
        inv_feat = x
        x = x.view(batch_size, -1, num_points)
        
        x1 = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
        x2 = F.adaptive_avg_pool1d(x, 1).view(batch_size, -1)
        x = torch.cat((x1, x2), 1)
        
        x = F.leaky_relu(self.bn1(self.linear1(x)), negative_slope=self.args.negative_slope)
        x = self.dp1(x)
        x = F.leaky_relu(self.bn2(self.linear2(x)), negative_slope=self.args.negative_slope)
        x = self.dp2(x)
        x = self.linear3(x)
        
        return x, inv_feat
