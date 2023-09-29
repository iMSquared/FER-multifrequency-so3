import torch
import torch.utils.data
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import os
import argparse
import time
import matplotlib; matplotlib.use('Agg')
import im2mesh.config
import im2mesh.data
from im2mesh.checkpoints import CheckpointIO
import random
from torch.backends import cudnn

import rotm_util_common as rmutil


def main(args: argparse.Namespace):

    # Compose configuration
    cfg  = im2mesh.config.load_config(
        path = args.config, 
        default_path = 'configs/default.yaml')
    device       = torch.device(args.device)        # Training device
    
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
    rot_configs = rmutil.init_rot_config(seed=args.rot_seed, dim_list=dim_list, device=device)
    rot_configs["constant_scale"] = args.rot_constant_scale
    cfg['model']['decoder_kwargs']['use_scale_params'] = args.use_scale_params
    cfg['model']['decoder_kwargs']['use_radial']       = args.use_radial
    cfg['model']['decoder_kwargs']['rot_configs']      = rot_configs
    cfg['model']['encoder_kwargs']['use_scale_params'] = args.use_scale_params
    cfg['model']['encoder_kwargs']['use_radial']       = args.use_radial
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
    num_workers  = args.num_workers                 # 
    out_dir      = cfg['training']['out_dir']       # 
    batch_size   = cfg['training']['batch_size']    # 64 default
    backup_every = cfg['training']['backup_every']  # Training steps
    max_iter     = args.max_iter
    learning_rate = args.lr
    print_every      = cfg['training']['print_every']
    checkpoint_every = cfg['training']['checkpoint_every']
    validate_every   = cfg['training']['validate_every']
    visualize_every  = cfg['training']['visualize_every']
    

    # Metric
    #   maximize: Save the model with the highest metric value
    #   minimize: Save the model with the lowest metric value
    model_selection_metric = cfg['training']['model_selection_metric']
    if cfg['training']['model_selection_mode'] == 'maximize':   
        model_selection_sign = 1
    elif cfg['training']['model_selection_mode'] == 'minimize':
        model_selection_sign = -1
    else:
        raise ValueError('model_selection_mode must be either maximize or minimize.')
    # Output directory
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)


    # Dataset
    train_dataset = im2mesh.config.get_dataset('train', cfg)
    val_dataset   = im2mesh.config.get_dataset('val', cfg)
    train_loader = torch.utils.data.DataLoader(
        dataset        = train_dataset, 
        batch_size     = batch_size, 
        num_workers    = num_workers, 
        shuffle        = True,
        collate_fn     = im2mesh.data.collate_remove_none,  # Custom: Prevents None tensor.
        worker_init_fn = im2mesh.data.seed_worker,          # Custom: Seed controlled
        generator      = g)
    val_loader = torch.utils.data.DataLoader(
        dataset        = val_dataset, 
        batch_size     = 10, 
        num_workers    = num_workers, 
        shuffle        = False,
        collate_fn     = im2mesh.data.collate_remove_none,  # Custom: Prevents None tensor.
        worker_init_fn = im2mesh.data.seed_worker,          # Custom: Seed controlled
        generator      = g)
    # For visualizations
    vis_loader = torch.utils.data.DataLoader(
        dataset        = val_dataset, 
        batch_size     = 12, 
        shuffle        = True,
        collate_fn     = im2mesh.data.collate_remove_none,
        worker_init_fn = im2mesh.data.seed_worker,
        generator      = g)
    data_vis = next(iter(vis_loader))


    # Model
    model: torch.nn.Module = im2mesh.config.get_model(
        cfg     = cfg, 
        device  = device, 
        dataset = train_dataset)

    # Initialize training
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    trainer   = im2mesh.config.get_trainer(model, optimizer, cfg, device=device)

    # Checkpoint module init
    checkpoint_io = CheckpointIO(out_dir, model=model, optimizer=optimizer, device=device)
    try:
        load_dict = checkpoint_io.load('model.pt')
    except FileExistsError:
        load_dict = dict()
    epoch_it = load_dict.get('epoch_it', -1)            # Num training epoch
    it       = load_dict.get('it', -1)                  # Num training steps
    metric_val_best = load_dict.get('loss_val_best',    # -inf if maximize. +inf if minimize.
                                    -model_selection_sign*np.inf) 

    # Print model
    nparameters = sum(p.numel() for p in model.parameters())
    print(model)
    print(f'Total number of parameters: {nparameters}')
    print(f'Current best validation metric ({model_selection_metric}): {metric_val_best:.8f}')
    logger = SummaryWriter(os.path.join(out_dir, 'logs'))


    # Epoch loop
    while True:
        if it > max_iter:
            break
        epoch_it += 1

        # Batch loop
        for batch in train_loader:
            it += 1
            loss = trainer.train_step(batch)
            logger.add_scalar('train/loss', loss, it)

            # Print output
            if print_every > 0 and (it % print_every) == 0:
                print(f'[Epoch {epoch_it:02d}] it={it:03d}, loss={loss:.4f}')

            # Visualize output
            if visualize_every > 0 and (it % visualize_every) == 0:
                print('Visualizing')
                trainer.visualize(data_vis)

            # Save checkpoint
            if (checkpoint_every > 0 and (it % checkpoint_every) == 0):
                print('Saving checkpoint')
                checkpoint_io.save(
                    filename = 'model.pt', 
                    epoch_it = epoch_it, 
                    it       = it,
                    loss_val_best = metric_val_best)
                    
            # Backup if necessary
            if (backup_every > 0 and (it % backup_every) == 0):
                print('Backup checkpoint')
                checkpoint_io.save(
                    filename = f'model_{it}.pt', 
                    epoch_it = epoch_it, 
                    it       = it,
                    loss_val_best = metric_val_best)
                
            # Run validation
            if validate_every > 0 and (it % validate_every) == 0:
                eval_dict = trainer.evaluate(val_loader)
                metric_val = eval_dict[model_selection_metric]
                print(f'Validation metric ({model_selection_metric}): {metric_val:.4f}')
                for k, v in eval_dict.items():
                    logger.add_scalar(f'val/{k}', v, it)
                # Save best model...
                if model_selection_sign * (metric_val - metric_val_best) > 0:
                    metric_val_best = metric_val
                    print(f'New best model (loss {metric_val_best:.4f})')
                    checkpoint_io.save(
                        filename = 'model_best.pt', 
                        epoch_it = epoch_it, 
                        it       = it,
                        loss_val_best = metric_val_best)

    # Finalize training
    print('Finalize training')
    checkpoint_io.save(
        filename = 'model.pt', 
        epoch_it = epoch_it, 
        it       = it,
        loss_val_best = metric_val_best)



if __name__=="__main__":

    # Arguments
    parser = argparse.ArgumentParser(description='Train a 3D reconstruction model.')
    parser.add_argument('--config',      type = str,                       help='Path to config file.')
    parser.add_argument('--device',      type = str,   default = "cuda:0", help="torch device")
    parser.add_argument('--exit-after',  type = int,   default = -1,       help='Checkpoint and exit after specified number of seconds with exit code 2.')
    parser.add_argument('--max-iter',    type = int,   default = 300000,   help='Max number of training iterations')
    parser.add_argument('--num_workers', type = int,   default = 16,       help='Max number of training iterations')
    parser.add_argument('--seed',        type = int,   default = 0,        help="random seed")
    parser.add_argument('--lr',          type = float, default = 0.0001,   help="Set custom learning rate")
    parser.add_argument('--override_c_dim', type = int, default= -1)

    # Ablations
    parser.add_argument('--use_scale_params',   action = "store_true", default = True,  help = "")
    parser.add_argument('--use_radial',         action = "store_true", default = False, help = "")
    parser.add_argument('--rot_seed',           type = int,            default = 0,     help = "")
    parser.add_argument('--rot_order',          type = str,            default = "1-2", help = "")
    parser.add_argument('--rot_constant_scale', action = "store_true",                  help = "")

    # Setup configurations
    args = parser.parse_args()
    main(args)
