import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from .ev_layers import *
from .utils.ev_util import *
import models.utils.rotm_util as rmutil


class ParamMakeFeature(nn.Module):
    def __init__(self, args, rot_configs, feature_axis=-2):
        super(ParamMakeFeature, self).__init__()
        self.args = args
        self.rot_configs = rot_configs
        self.feature_axis = feature_axis

        feat_sum = np.sum(np.array(args.dim_list) * 2 + 1)
        self.feature_mixing = nn.Sequential(
            nn.Linear(3, feat_sum, bias=True),
            nn.ReLU(),
            nn.Linear(feat_sum, feat_sum, bias=True),
            nn.ReLU(),
            nn.Linear(feat_sum, feat_sum, bias=True),
            )
        

    def forward(self, x):
        randRm = rmutil.rand_matrix_torch(x.size(0))
        randRm = torch.where(torch.randint(0, 5, size=(x.size(0),1,1))>=1,randRm, torch.eye(3)).to(x.device)
        for _ in range(x.dim() + 1 - randRm.dim()):
            randRm = randRm.unsqueeze(-3)
        randRminv = randRm.transpose(-2,-1)
        x = torch.einsum('...ij,...j', randRminv, x.transpose(self.feature_axis, -1))
        x = self.feature_mixing(x).transpose(self.feature_axis, -1)
        x = rmutil.apply_rot(x, randRm, self.rot_configs, feature_axis=self.feature_axis)
        return x




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


class EVPointNetEncoder(nn.Module):
    def __init__(self, args, global_feat=True, feature_transform=False, channel=3, rot_configs=None):
        super(EVPointNetEncoder, self).__init__()
        self.rot_configs = rot_configs
        self.args = args
        self.n_knn = 20

        if args.psi_scale_type >= 2:
            self.make_feature = MakeHDFeature(self.rot_configs, args, nfeat=3)
        else:
            self.scale_params = nn.Parameter(data=torch.Tensor(torch.ones(len(self.rot_configs['dim_list']))), requires_grad=True)
            self.make_feature = lambda x: make_features(x, self.rot_configs, self.scale_params, 2)
        feat_dim = self.args.feat_dim
        self.conv_pos = EVLinearLeakyReLU(3, 64//feat_dim, args=args)
        self.conv1 = EVLinearLeakyReLU(64//feat_dim, 64//feat_dim, args=args)
        self.conv2 = EVLinearLeakyReLU(64//feat_dim*2, 128//feat_dim, args=args)
        
        self.conv3 = EVNLinear(128//feat_dim, args.embedding_size//feat_dim)
        self.bn3 = EVNNonLinearity(args.embedding_size//feat_dim, args=args)

        self.std_feature = EVNStdFeature(args.embedding_size//feat_dim*2, normalize_frame=False, args=args)
        
        self.global_feat = global_feat
        self.feature_transform = feature_transform
        
        self.fstn = EVSTNkd(args, d=64//feat_dim, feat_dim=feat_dim)

    def forward(self, x):
        B, D, N = x.size()

        x = x.unsqueeze(1)
        feat = get_graph_feature_cross(x, k=self.n_knn)
        feat = self.make_feature(feat)
        x = self.conv_pos(feat)
        x = torch.mean(x, -1) # (NB, NF, 3, NP)

        x = self.conv1(x)
        x_global = self.fstn(x).unsqueeze(-1).repeat(1,1,1,N)
        x = torch.cat((x, x_global), 1)
        
        x = self.conv2(x)
        if self.args.batch_norm:
            x = self.bn3(self.conv3(x))
        else:
            x = self.conv3(x)
        
        x_mean = x.mean(dim=-1, keepdim=True).expand(x.size())
        x = torch.cat((x, x_mean), 1)
        x, trans = self.std_feature(x)
        inv_feat = x
        x = x.view(B, -1, N)
        
        x = torch.max(x, -1, keepdim=False)[0]
        
        trans_feat = None
        return x, trans, trans_feat, inv_feat


class EV_POINTNET_CLS(nn.Module):
    def __init__(self, args, num_class=40):
        super(EV_POINTNET_CLS, self).__init__()
        self.args = args
        channel = 3

        rot_configs = rmutil.init_rot_config(args.seed, args.rot_order, args.rot_type)
        rot_configs['constant_scale'] = 0

        feat_dim = np.sum(2*np.array(rot_configs['dim_list'])+1)
        feat_dim = np.minimum(feat_dim, 20)

        self.args.dim_list = rot_configs['dim_list']
        self.args.feat_dim = feat_dim

        self.feat = EVPointNetEncoder(args, global_feat=True, feature_transform=True, channel=channel, rot_configs=rot_configs)
        self.fc1 = nn.Linear(args.embedding_size//feat_dim*2*feat_dim, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_class)
        self.dropout = nn.Dropout(p=0.4)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.relu = nn.ReLU()

    def forward(self, x):
        x, trans, trans_feat, inv_feat = self.feat(x)
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.dropout(self.fc2(x))))
        x = self.fc3(x)
        return x, inv_feat