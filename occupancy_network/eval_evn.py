import argparse
import os
import pandas as pd
import torch
import torch.utils.data
import numpy as np
from tqdm import tqdm
import im2mesh.config
import im2mesh.data
from im2mesh.checkpoints import CheckpointIO
import random
from torch.backends import cudnn

import rotm_util_common as rmutil



def main(args: argparse.Namespace):

    # Compose configuration
    cfg = im2mesh.config.load_config(
        args.config, 
        'configs/default.yaml')
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
    cfg['model']['decoder_kwargs']['use_radial']       = args.use_radial
    cfg['model']['decoder_kwargs']['rot_configs']      = rot_configs
    cfg['model']['encoder_kwargs']['use_scale_params'] = args.use_scale_params
    cfg['model']['encoder_kwargs']['use_radial']       = args.use_radial
    cfg['model']['encoder_kwargs']['rot_configs']      = rot_configs


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
    out_file = os.path.join(out_dir, 'eval_full.pkl')
    out_file_class = os.path.join(out_dir, 'eval.csv')

    # Dataset
    dataset = im2mesh.config.get_dataset('test', cfg, return_idx=True)
    test_loader = torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=False,
        collate_fn=im2mesh.data.collate_remove_none,
        worker_init_fn=im2mesh.data.seed_worker)

    # Model
    model = im2mesh.config.get_model(cfg, device=device, dataset=dataset)
    # Checkpoint module init + load checkpoint
    checkpoint_io = CheckpointIO(out_dir, model=model)
    try:
        checkpoint_io.load(cfg['test']['model_file'])
    except FileExistsError:
        print('Model file does not exist. Exiting.')
        exit()
    # Trainer (for evaluation)
    trainer = im2mesh.config.get_trainer(model, None, cfg, device=device)


    # Print model
    nparameters = sum(p.numel() for p in model.parameters())
    print(model)
    print('Total number of parameters: %d' % nparameters)

    # Evaluate
    model.eval()
    eval_dicts = []   
    print('Evaluating networks...')

    # Handle each dataset separately
    for it, data in enumerate(tqdm(test_loader)):
        if data is None:
            print('Invalid data.')
            continue
        # Get index etc.
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
            'modelname':modelname,
        }
        eval_dicts.append(eval_dict)
        eval_data = trainer.eval_step(data)
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

    # Print results
    print(eval_df_class)




if __name__=="__main__":

    # Arguments
    parser = argparse.ArgumentParser(description='Evaluate mesh algorithms.')
    parser.add_argument('--config', type = str,                     help = "Path to config file.")
    parser.add_argument('--device', type = str, default = "cuda:0", help = "torch device")
    parser.add_argument('--seed',   type = int, default = 0,        help = "")

    # Ablations
    parser.add_argument('--use_scale_params',   action = "store_true", default = True,  help = "")
    parser.add_argument('--use_radial',         action = "store_true", default = False, help = "")
    parser.add_argument('--rot_seed',           type   = int,          default = 0,     help = "")
    parser.add_argument('--rot_order',          type   = str,          default = "1-2", help = "")
    parser.add_argument('--rot_constant_scale', action = "store_true",                  help = "")

    # Get configuration and basic arguments
    args = parser.parse_args()
    main(args)

