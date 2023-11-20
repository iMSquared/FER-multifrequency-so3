import open3d as o3d
import numpy as np
import torch
from torch import nn
import argparse
import einops
from torch.utils.data import Dataset
from tqdm import tqdm
import datetime
import os
import glob

from torch.utils.tensorboard import SummaryWriter

import models.utils.rotm_util as rmutil
# from ev_layers import *
# import ev_util as eutil
import models.graph_onet as gonet
import models.graph_onet_FER as gonet_fer

try:
    import vessl
    vessl.init()
    vessl_on = True
except:
    vessl_on = False


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=1e-3, help="Input batch size.")
    parser.add_argument("--batch_query_size", type=int, default=1024, help="Input batch size.")
    parser.add_argument("--npoint", type=int, default=300, help="Input batch size.")
    parser.add_argument("--nepochs", type=int, default=1000, help="Input batch size.")
    parser.add_argument("--embedding_size", type=int, default=32, help="Input batch size.")
    parser.add_argument("--rot_order", type=str, default='1-2', help="Input batch size.")
    parser.add_argument("--seed", type=int, default=1, help="Input batch size.")
    parser.add_argument("--batch_size", type=int, default=20, help="Input batch size.")
    parser.add_argument("--train_rot", type=str, default='aligned', help="Input batch size.")
    parser.add_argument("--psi_scale_type", type=int, default=3)
    parser.add_argument("--negative_slope", type=float, default=0.2)
    parser.add_argument("--model_type", type=str, default='fer')
    parser.add_argument("--vn_nonlinearity", type=str, default='fer')
    parser.add_argument("--hn_nonlinearity", type=str, default='vn')
    parser.add_argument("--residual", type=int, default=0)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--base_dim", type=int, default=8)
    
    args = parser.parse_args()

    # %%
    #  create dataset

    def create_ds_pool(obj_path, npcd, nqpnt):
        mesh = o3d.io.read_triangle_mesh(obj_path)
        mesh.compute_vertex_normals()
        aabb = mesh.get_axis_aligned_bounding_box()
        mesh.translate(-0.5*(aabb.max_bound + aabb.min_bound))
        scale = 0.55*np.max(aabb.max_bound - aabb.min_bound)
        mesh.scale(1/scale, np.zeros(3))

        input_pnts = np.array(mesh.sample_points_uniformly(npcd).points).astype(np.float32)

        tmesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
        scene = o3d.t.geometry.RaycastingScene()
        tmesh_id = scene.add_triangles(tmesh)

        # occpancy query
        qps = mesh.sample_points_uniformly(int(nqpnt*0.95))
        qps = np.concatenate([np.array(qps.points), np.random.uniform(-1,1,size=(int(nqpnt*0.05),3))], 0)
        qps = qps + np.random.normal(size=qps.shape) * 0.005
        qps = o3d.core.Tensor(qps,
                            dtype=o3d.core.Dtype.Float32)
        ans = scene.compute_occupancy(qps)

        # pcd = o3d.geometry.PointCloud()
        # pcd.points = o3d.utility.Vector3dVector(qps.numpy()[ans.numpy()==1])
        # o3d.visualization.draw_geometries([pcd])

        return input_pnts.astype(np.float32), qps.numpy().astype(np.float32), ans.numpy()

    class OccDataLoader(Dataset):
        def __init__(self, npcd, nqpnts, data_len=10000):
            dir_list = glob.glob("data/stanford/*.obj")
            self.data_len = data_len

            self.ds_list = []
            for dl in dir_list:
                self.ds_list.append(create_ds_pool(dl, npcd, nqpnts))

        def __len__(self):
            return self.data_len

        def __getitem__(self, index):
            idx = np.random.randint(0, len(self.ds_list))
            pcd, qps, occ = self.ds_list[idx]
            idx_pcd = np.random.randint(0, pcd.shape[0], size=(args.npoint,))
            idx_qps = np.random.randint(0, qps.shape[0], size=(args.batch_query_size,))
            pcd_res, qps_res = pcd[idx_pcd], qps[idx_qps]
            return pcd_res, qps_res, occ[idx_qps]

    train_dataset = torch.utils.data.DataLoader(OccDataLoader(20000, 1000000), batch_size=args.batch_size, shuffle=True)
    eval_dataset = torch.utils.data.DataLoader(OccDataLoader(20000, 100000, 1000), batch_size=args.batch_size, shuffle=False)

# %%

class OccNet(nn.Module):
    def __init__(self, args):
        super(OccNet, self).__init__()
        self.enc = gonet.hn_DGCNN_spatial_fps_hier(k=args.k, c_dim=args.embedding_size)
        self.dec = gonet.hn_DGCNN_decoder_hier2(k=args.k, c_dim=args.embedding_size)
    
    def forward(self, spnts, qpnts):
        encoded = self.enc(spnts)
        qres = self.dec(qpnts, encoded)
        return qres

class OccNetFer(nn.Module):
    def __init__(self, args):
        super(OccNetFer, self).__init__()
        rot_configs = rmutil.init_rot_config(seed=args.seed, dim_list=args.rot_order, rot_type='custom')
        self.enc = gonet_fer.hn_DGCNN_spatial_fps_hier(args, rot_configs, k=args.k, c_dim=args.embedding_size)
        self.dec = gonet_fer.hn_DGCNN_decoder_hier2(args, rot_configs, k=args.k, c_dim=args.embedding_size)
    
    def forward(self, spnts, qpnts):
        encoded = self.enc(spnts)
        qres = self.dec(qpnts, encoded)
        return qres

if __name__ == '__main__':

    if args.model_type == 'fer':
        model = OccNetFer(args=args).to('cuda')
    else:
        model = OccNet(args=args).to('cuda')

    # %%
    # define loss
    def cal_loss(yp, yt):
        yp = torch.sigmoid(yp)
        yp = yp.clip(1e-5, 1-1e-5)
        loss = yt*torch.log(yp) + (1-yt)*torch.log((1-yp))
        return -torch.mean(loss)

    # %%
    # start training
    now = datetime.datetime.now() # current date and time
    date_time = now.strftime("%m%d%Y-%H%M%S" + '_' + args.rot_order)
    if vessl_on:
        logs_dir = os.path.join('/output', date_time)
    else:
        logs_dir = os.path.join('logs', date_time)
    writer = SummaryWriter(logs_dir)
    
    fn = os.path.basename(__file__)
    os.system("cp -r {0} {1}".format(__file__, os.path.join(logs_dir, fn)))

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criteria = cal_loss
    best_iou_so3 = 0
    best_iou_aligned = 0
    for ep in range(args.nepochs):
        model.train()
        for point_cloud, points, labels in tqdm(train_dataset, total=len(train_dataset), smoothing=0.9):
            point_cloud = point_cloud.cuda()
            points = points.cuda()
            labels = labels.cuda()
            if args.train_rot == 'so3':
                qrand = rmutil.qrand((points.shape[0],)).cuda()
                points = rmutil.qaction(qrand[:,None], points)
                point_cloud = rmutil.qaction(qrand[:,None], point_cloud)
            output = model(point_cloud, points)    
            loss = criteria(output, labels)
            loss.backward()
            optimizer.step()
            for param in model.parameters():
                param.grad = None

        def eval(test_rot):
            model.eval()
            total_loss = 0
            itr_idx = 0
            npositive = 0
            ntotal = 0
            with torch.no_grad():
                for point_cloud, points, labels in eval_dataset:
                    itr_idx += 1
                    point_cloud = point_cloud.cuda()
                    points = points.cuda()
                    labels = labels.cuda()
                    if test_rot == 'so3':
                        qrand = rmutil.qrand((points.shape[0],)).cuda()
                        points = rmutil.qaction(qrand[:,None], points)
                        point_cloud = rmutil.qaction(qrand[:,None], point_cloud)
                    output = model(point_cloud, points)
                    loss = criteria(output, labels)
                    total_loss += loss
                    npositive += torch.sum(torch.logical_and(output>0.5, labels>0.5))
                    ntotal += torch.sum(torch.logical_or(output>0.5, labels>0.5))
            return total_loss/itr_idx, npositive/ntotal

        if ep%3==0:
            loss_so3, iou_so3 = eval('so3')
            loss_aligned, iou_aligned = eval('aligned')

            if iou_so3>best_iou_so3:
                best_iou_so3 = iou_so3
            if iou_aligned>best_iou_aligned:
                best_iou_aligned = iou_aligned
                torch.save({"args":args, "static_dict":model.state_dict()}, os.path.join(logs_dir, "best.pth"))
                print(f'best iou {best_iou_aligned}')
            print(f"{ep}, iou_so3:{iou_so3}, iou_aligned:{iou_aligned}, best_iou_so3:{best_iou_so3}, best_iou_aligned:{best_iou_aligned}")

            writer.add_scalar('loss_so3', loss_so3, ep)
            writer.add_scalar('iou_so3', iou_so3, ep)
            writer.add_scalar('iou_so3_best', best_iou_so3, ep)
            writer.add_scalar('loss_aligned', loss_aligned, ep)
            writer.add_scalar('iou_aligned', iou_aligned, ep)
            writer.add_scalar('iou_aligned_best', best_iou_aligned, ep)

            if vessl_on:
                base_name = "PT/"
                log_dict = {'loss_so3':loss_so3, "iou_so3":iou_so3, "iou_so3_best":best_iou_so3, "loss_aligned":loss_aligned, "iou_aligned":iou_aligned, "iou_aligned_best":best_iou_aligned}
                log_dict = {base_name+k: log_dict[k] for k in log_dict}
                vessl.log(step=ep, payload=log_dict)

