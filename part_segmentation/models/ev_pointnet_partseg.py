import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
import torch.nn.functional as F
from .ev_layers import *
from .utils.ev_util import get_graph_feature_cross

import models.utils.rotm_util as rmutil

class EVSTNkd(nn.Module):
    def __init__(self, args, d=64, feat_dim=3):
        super(EVSTNkd, self).__init__()
        self.args = args
        
        self.conv1 = EVLinearLeakyReLU(d, 64//3, args=args)
        self.conv2 = EVLinearLeakyReLU(64//3, 128//3, args=args)
        self.conv3 = EVLinearLeakyReLU(128//3, 1024//feat_dim, args=args)

        self.fc1 = EVLinearLeakyReLU(1024//feat_dim, 512//feat_dim, args=args)
        self.fc2 = EVLinearLeakyReLU(512//feat_dim, 256//feat_dim, args=args)
        
        self.fc3 = EVNLinear(256//feat_dim, d)
        self.d = d

    def forward(self, x):
        batchsize = x.size()[0]
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = torch.mean(x, -1)

        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        
        return x

class EV_PointNet_PSEG(nn.Module):
    def __init__(self, args, num_part=50):
        super(EV_PointNet_PSEG, self).__init__()
        self.args = args
        self.k = args.k
        self.num_part = num_part

        rot_configs = rmutil.init_rot_config(args.seed, args.rot_order, args.rot_type)
        rot_configs['constant_scale'] = 0
        self.rot_configs = rot_configs

        feat_dim = np.sum(2*np.array(rot_configs['dim_list'])+1)
        feat_dim = np.minimum(feat_dim, 20)

        self.args.dim_list = rot_configs['dim_list']
        self.args.feat_dim = feat_dim
        self.make_feature = MakeHDFeature(self.rot_configs, args, 3)
        
        self.conv_pos = EVLinearLeakyReLU(3, 64//feat_dim, args=args)
        self.conv1 = EVLinearLeakyReLU(64//feat_dim, 64//feat_dim, args=args)
        self.conv2 = EVLinearLeakyReLU(64//feat_dim, 128//feat_dim, args=args)
        self.conv3 = EVLinearLeakyReLU(128//feat_dim, 128//feat_dim, args=args)
        self.conv4 = EVLinearLeakyReLU(128//feat_dim*2, 512//feat_dim, args=args)
        
        self.conv5 = EVNLinear(512//feat_dim, 2048//feat_dim)
        self.bn5 = EVNNonLinearity(2048//feat_dim, args=args)
        
        self.std_feature = EVNStdFeature(2048//feat_dim*2, normalize_frame=False, args=args)
        
        self.fstn = EVSTNkd(args, d=128//feat_dim, feat_dim=feat_dim)

        concat_dim = (2048//feat_dim*2*feat_dim+16) + feat_dim*(64//feat_dim + 128//feat_dim + 128//feat_dim + 512//feat_dim) + 2048//feat_dim*2 * feat_dim
        
        self.convs1 = torch.nn.Conv1d(concat_dim, 256, 1)
        self.convs2 = torch.nn.Conv1d(256, 256, 1)
        self.convs3 = torch.nn.Conv1d(256, 128, 1)
        self.convs4 = torch.nn.Conv1d(128, num_part, 1)
        self.bns1 = nn.BatchNorm1d(256)
        self.bns2 = nn.BatchNorm1d(256)
        self.bns3 = nn.BatchNorm1d(128)

    def forward(self, point_cloud, label):
        B, D, N = point_cloud.size()

        point_cloud = point_cloud.unsqueeze(1)
        feat = get_graph_feature_cross(point_cloud, k=self.k)
        feat = self.make_feature(feat)
        point_cloud = self.conv_pos(feat)
        point_cloud = torch.mean(point_cloud, -1)

        out1 = self.conv1(point_cloud)
        out2 = self.conv2(out1)
        out3 = self.conv3(out2)
        
        net_global = self.fstn(out3).unsqueeze(-1).repeat(1,1,1,N)
        net_transformed = torch.cat((out3, net_global), 1)

        out4 = self.conv4(net_transformed)
        out5 = self.bn5(self.conv5(out4))
        
        out5_mean = out5.mean(dim=-1, keepdim=True).expand(out5.size())
        out5 = torch.cat((out5, out5_mean), 1)
        out5, trans = self.std_feature(out5) # B, 4096
        out5 = out5.view(B, -1, N)
        
        out_max = torch.max(out5, -1, keepdim=False)[0]

        out_max = torch.cat([out_max,label.squeeze(1)],1)
        expand = out_max.view(-1, 2048//self.args.feat_dim*2*self.args.feat_dim+16, 1).repeat(1, 1, N)
        
        out1234 = torch.cat((out1, out2, out3, out4), dim=1)
        out1234 = torch.einsum('bijm,bjkm->bikm', out1234, trans).contiguous().view(B, -1, N)
        
        concat = torch.cat([expand, out1234, out5], 1)
        
        net = F.relu(self.bns1(self.convs1(concat)))
        net = F.relu(self.bns2(self.convs2(net)))
        net = F.relu(self.bns3(self.convs3(net)))
        net = self.convs4(net)
        
        return net

