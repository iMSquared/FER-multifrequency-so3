import argparse
import os
import pandas as pd
import torch
import torch.utils.data
import numpy as np
import numpy.typing as npt
from tqdm import tqdm
import im2mesh.config
import im2mesh.data
import im2mesh.common
from im2mesh.checkpoints import CheckpointIO
import random
import torch.backends.cudnn as cudnn

from registration.so3 import log, exp
from registration.transforms import apply_rot
from registration.register_utils import *
from typing import Optional
import matplotlib.pyplot as plt
from custom_metric_utils import visualize_point_cloud_render, visualize_registration

import rotm_util_common as rmutil



def optimize_so3_logarithm(feat1, feat2, rot_configs, R_gt) -> torch.Tensor:
    """No batch dim

    Args:
        feat1 (_type_): _description_
        feat2 (_type_): _description_
        rot_configs (_type_): _description_

    Returns:
        _type_: _description_
    """
    
    def objective_func1(theta, feat1, feat2, rot_configs):
        dim_list = rot_configs["dim_list"]
        feat1_dims = []
        feat2_dims = []
        dim_start = 0
        for dim in dim_list:
            # Split dim
            dim_end = dim_start + dim*2+1
            feat1_dim = feat1[:,dim_start:dim_end]
            feat2_dim = feat2[:,dim_start:dim_end]
            dim_start += dim*2+1
            # Aggregate
            feat1_dims.append(feat1_dim)
            feat2_dims.append(feat2_dim)
        # exp(theta) -> D(R)
        Rs = []
        rm_3d = exp(theta)
        for dim in dim_list:
            if dim == 1:
                Rs.append(rm_3d)
            else:                
                rm_nd = rmutil.custom_rotmV2(dim, rm_3d, rot_configs["D_basis"])
                Rs.append(rm_nd)
        # Equation to minimize
        norm_list = []
        for dim, m_Rn, feat1_dim, feat2_dim in zip(dim_list, Rs, feat1_dims, feat2_dims):
            feat1_dim_rotated = torch.einsum("nij,fj->nfi", m_Rn, feat1_dim)
            diff = feat1_dim_rotated - feat2_dim.unsqueeze(0)
            frob_norm = torch.linalg.matrix_norm(diff)
            norm_list.append(frob_norm)
        # Aggregate cost
        cost = sum(norm_list)
        # print(f"dim {0}: {norm_list[0]}")
        # print(f"dim {1}: {norm_list[1]}")
        return cost
    

    # CEM iterations
    total_n = 10000  
    beta_theta = torch.Tensor(np.random.uniform(low=-np.pi, high=np.pi, size=(total_n, 3))).to(feat1.device)
    itr_no = 10
    cem_ratio=0.02
    for i in range(itr_no):
        # Loss
        loss = objective_func1(beta_theta, feat1, feat2, rot_configs)
        # Beta update        
        topkargs = torch.argsort(loss)[:int(total_n*cem_ratio)] # Lower the better
        mean = torch.mean(beta_theta[topkargs], dim=0)
        std  = torch.std(beta_theta[topkargs], dim=0).clip(1e-9)
        beta_theta = mean + torch.randn(size=(total_n, 3)).to(std.device) * std

    # Pick top 1
    loss = objective_func1(beta_theta, feat1, feat2, rot_configs)
    top_arg = torch.argsort(loss)[0]
    m_so3 = exp(beta_theta[top_arg])

    return m_so3



def main(args):

    # Compose configuration
    cfg = im2mesh.config.load_config(
        path = args.config, 
        default_path = 'configs/registration/default_registration.yaml')
    device = torch.device(args.device)

    # Seed control
    SEED = args.seed
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    np.random.seed(SEED)
    cudnn.benchmark = False
    cudnn.deterministic = True
    random.seed(SEED)
    g = torch.Generator()
    g.manual_seed(SEED)

    # Compose EVN ablation from command lines
    stdim    = int(args.rot_order.split('-')[0])
    edim     = int(args.rot_order.split('-')[1])
    dim_list = list(range(stdim, edim+1))
    # TODO: Fix this...
    rot_configs = rmutil.init_rot_config(seed=args.rot_seed, dim_list=dim_list, device=device) 
    rot_configs["constant_scale"] = args.rot_constant_scale
    cfg['model']['decoder_kwargs']['use_scale_params'] = args.use_scale_params
    cfg['model']['decoder_kwargs']['rot_configs']      = rot_configs
    cfg['model']['encoder_kwargs']['use_scale_params'] = args.use_scale_params
    cfg['model']['encoder_kwargs']['rot_configs']      = rot_configs
    if args.override_c_dim > 0:
        c_dim = args.override_c_dim
        cfg['model']['c_dim'] = c_dim
        cfg['model']['encoder_kwargs']['hidden_dim'] = 2*c_dim

    # Reset seed again
    SEED = args.seed
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    np.random.seed(SEED)
    cudnn.benchmark = False
    cudnn.deterministic = True
    random.seed(SEED)
    g = torch.Generator()
    g.manual_seed(SEED)


    # Shorthands
    out_dir = cfg['training']['out_dir']
    out_file = os.path.join(out_dir, 'eval_registration_180.pkl')
    out_file_class = os.path.join(out_dir, 'eval_registration_180.csv')

    # Dataset
    dataset = im2mesh.config.get_dataset('test', cfg, return_idx=True)
    test_loader = torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=False,
        collate_fn=im2mesh.data.collate_remove_none,
        worker_init_fn=im2mesh.data.seed_worker,
        generator=g)
    
    # Model
    model = im2mesh.config.get_model(cfg, device=device, dataset=dataset)
    # Checkpoint module init + load checkpoint
    checkpoint_io = CheckpointIO(out_dir, model=model)
    try:
        checkpoint_io.load(cfg['test']['model_file'])
    except FileExistsError:
        print('Model file does not exist. Exiting.')
        exit()


    # Evaluate
    model.eval()
    eval_dicts = []   
    print('Evaluating networks...')

    # Handle each dataset separately
    for it, data in enumerate(tqdm(test_loader)):
        with torch.no_grad():
            # Parse items from dataloader
            for key, value in data.items():
                if isinstance(value, torch.Tensor):
                    data[key] = value.to(device)
            # Input to numpy for visualization
            np_inputs1 = data["inputs"].cpu().squeeze(0).numpy()
            np_inputs2 = data["inputs2_so3"].cpu().squeeze(0).numpy()
            # Encode
            out_1 = model.encode_inputs(data["inputs"])
            out_2 = model.encode_inputs(data["inputs2_so3"])

        # Predict R (p1 -> p2)
        R_gt   = data["R_gt"]
        R_pred = optimize_so3_logarithm(
            out_1.detach().squeeze(0), 
            out_2.detach().squeeze(0), 
            rot_configs,
            R_gt)

        # To numpy
        np_R_gt   = R_gt.cpu().squeeze(0).detach().numpy()
        np_R_pred = R_pred.cpu().squeeze(0).detach().numpy()

        # Rotate back
        np_inputs2_registered = np_inputs2@np_R_pred
        # visualize_point_cloud([out_1[0,:,:3].cpu().detach().numpy(), 
        #                        out_2[0,:,:3].cpu().detach().numpy(),
        #                        out_2[0,:,:3].cpu().detach().numpy()@np_R_pred])

        # Metric
        angle_diff_degree = angle_diff_func(R_pred, R_gt)
        angle_diff_degree = angle_diff_degree.item()
        chamfer_dist = im2mesh.common.chamfer_distance(torch.Tensor(np.expand_dims(np_inputs1, axis=0)).to(device), 
                                                       torch.Tensor(np.expand_dims(np_inputs2_registered, axis=0)).to(device)).item()
        
        
        # # # Log
        # print(f"Pred: \n{np_R_pred}")
        # print(f"GT  : \n{np_R_gt}")
        # print(f"angle_difference: {angle_diff_degree}")
        # print(f"chamfer_distance: {chamfer_dist}")
        # # visualize_registration(np_inputs1, np_inputs2, np_inputs2_registered)
        # visualize_point_cloud_render([np_inputs1, np_inputs2_registered])

        idx = data['idx'].item()
        try:
            model_dict = dataset.get_model_dict(idx)
        except AttributeError:
            model_dict = {'model': str(idx), 'category': 'n/a'}
        modelname = model_dict['model']
        category_id = model_dict['category']
        try:
            category_name = dataset.metadata[category_id].get('name', 'n/a')
        except AttributeError:
            category_name = 'n/a'
        eval_dict = {
            'idx': idx,
            'class id': category_id,
            'class name': category_name,
            'modelname': modelname,
        }
        eval_data = {
            'angle difference': angle_diff_degree,
            'chamfer distance': chamfer_dist,
        }
        eval_dicts.append(eval_dict)
        eval_dict.update(eval_data)

    # Create pandas dataframe and save
    eval_df = pd.DataFrame(eval_dicts)
    eval_df.set_index(['idx'], inplace=True)
    eval_df.to_pickle(out_file)

    # Create CSV file  with main statistics
    eval_df_class = eval_df.groupby(by=['class name']).mean(numeric_only=True)
    eval_df_class.loc['class_mean'] = eval_df_class.mean()
    eval_df_class.loc['instance_mean'] = eval_df.mean(numeric_only=True)
    eval_df_class.to_csv(out_file_class)



if __name__=="__main__":

    # Arguments
    parser = argparse.ArgumentParser(description='Evaluate mesh algorithms.')
    parser.add_argument('--config', type=str, default="configs/registration/evn_pointnet_resnet_registration.yaml", help='Path to config file.')
    parser.add_argument('--device', type=str, default="cuda:0", help="torch device")
    parser.add_argument('--seed',   type=int, default=0,        help="random seed")

    # Ablations
    parser.add_argument('--use_scale_params',   action="store_true", default=True,  help="")
    parser.add_argument('--rot_seed',           type  =int,          default=0,     help="")
    parser.add_argument('--rot_order',          type  =str,          default="1-2", help="")
    parser.add_argument('--rot_constant_scale', action="store_true",                help="")
    parser.add_argument('--override_c_dim',     type  =int,          default=-1,    help="")

    # Get configuration and basic arguments
    args = parser.parse_args()
    main(args)