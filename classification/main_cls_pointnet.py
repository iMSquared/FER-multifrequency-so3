import argparse
import os, sys

parser = argparse.ArgumentParser(description='Point Cloud Recognition using PointNet backbone')
parser.add_argument('--rot_type', type=str, default='custom')
parser.add_argument('--embedding_size', type=int, default=1024)
parser.add_argument('--activation', type=int, default=0)
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--rot_order', type=str, default='1-2')
parser.add_argument('--psi_scale_type', type=int, default=3)
parser.add_argument('--negative_slope', type=float, default=0.0)
parser.add_argument('--batch_norm', type=int, default=6)
parser.add_argument('--model', type=str, default='evnet', metavar='N',
                    choices=['original', 'vn', 'evnet'])
parser.add_argument('--batch-size', type=int, default=32, metavar='batch_size',
                    help='Size of batch)')
parser.add_argument('--epochs', type=int, default=200, metavar='N',
                    help='number of episode to train ')
parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                    help='learning rate (default: 0.001, 0.1 if using sgd)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum (default: 0.9)')
parser.add_argument('--wd', type=float, default=1e-4, metavar='WD',
                    help='weight decay')
parser.add_argument('--num-points', type=int, default=1024,
                    help='num of points to use')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='dropout rate')
parser.add_argument('--emb-dims', type=int, default=1024, metavar='N',
                    help='Dimension of embeddings')
parser.add_argument('--k', type=int, default=20, metavar='N',
                    help='Num of nearest neighbors to use')
parser.add_argument('--rot', type=str, default='so3', metavar='N',
                    choices=['aligned', 'z', 'so3'],
                    help='Rotation augmentation to input data')
parser.add_argument('--rot-test', type=str, default='so3', metavar='N',
                    choices=['aligned', 'z', 'so3'],
                    help='Rotation augmentation to input data during testing')
parser.add_argument('--data-dir', metavar='DATADIR', type=str, default=os.path.join(os.path.dirname(__file__),'data'),
                    help='data dir to load datasets')
parser.add_argument('--save-dir', metavar='SAVEDIR', type=str, default=os.path.join(os.path.dirname(__file__),'results'),
                    help='dir to save logs and model checkpoints')
parser.add_argument('--checkinfo', action='store_true', 
                    help='only check the information of the model')
args = parser.parse_args()

import time
import numpy as np
import sklearn.metrics as metrics
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from data import ModelNet40

sys.path.insert(0, os.path.dirname(__file__))

import models
import utils
import models.utils.rotm_util as rmutil

try:
    import vessl
    vessl_on = True
    vessl.init()
except:
    vessl_on = False

torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
np.random.seed(args.seed)

log_string = utils.configure_logging(args.save_dir, 'cls')
epoch_string = utils.configure_logging(args.save_dir, 'cls', 'log')

def main():
    epoch_string(str(args))
    num_class = 40

    #Try to load models
    criterion = utils.cal_loss
    if args.model == 'original':
        model = models.PointNet_CLS(args, num_class=num_class)
        criterion = utils.cal_pointnet_loss
    elif args.model == 'vn':
        model = models.VN_PointNet_CLS(args, num_class=num_class)
    elif args.model == 'evnet':
        criterion = lambda x,y: utils.cal_loss(x, y)
        model = models.EV_POINTNET_CLS(args, num_class=num_class)
    else:
        raise Exception("Not implemented")

    train_loader = DataLoader(ModelNet40(data_dir=args.data_dir, partition='train', num_points=args.num_points), num_workers=8, batch_size=args.batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(ModelNet40(data_dir=args.data_dir, partition='test', num_points=args.num_points), num_workers=8, batch_size=args.batch_size, shuffle=True, drop_last=False)

    log_string(f'trainloader: {len(train_loader.dataset)}, test_loader: {len(test_loader.dataset)}')

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    # model = nn.DataParallel(model.to(device))
    # log_string("Let's use {} GPUs!".format(torch.cuda.device_count()))

    log_string('use adam')
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=args.wd)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.7)

    ## begin training or testing
    start_epoch = 0
    best_test_acc = 0

    checkpoint = utils.load_checkpoint(args)
    if checkpoint is not None:
        model.load_state_dict(checkpoint['state_dict'])
        if args.test is None:
            start_epoch = checkpoint['epoch'] + 1
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler'])
            best_test_acc = checkpoint['best_test_acc']
        log_string('checkpoint loaded successfully')
    else:
        log_string('no checkpoint loaded')

    saveID = None
    print_freq = len(train_loader) // 10
    for epoch in range(start_epoch, args.epochs):
        lr = optimizer.param_groups[0]['lr']
        train_loss = 0.0
        count = 0.0
        model.train()
        train_pred = []
        train_true = []
        for i, (data, label) in enumerate(train_loader):
            trot = None
            if args.rot == 'z':
                zang = torch.rand(data.shape[0]) * 2*np.pi
                trot = rmutil.aa2q(torch.stack([torch.zeros_like(zang), torch.zeros_like(zang), zang], -1)).to(device)
            elif args.rot == 'so3':
                trot = rmutil.qrand((data.shape[0],)).to(device)

            data, label = data.to(device), label.to(device, dtype=torch.long).squeeze()
            if trot is not None:
                data = rmutil.qaction(trot[...,None,:], data)
            data = data.permute(0, 2, 1)
            batch_size = data.size()[0]
            optimizer.zero_grad()
            logits = model(data)
            loss = criterion(logits, label)
            loss.backward()
            optimizer.step()
            if args.model in ['original', 'bipointnet', 'evnet']:
                preds = logits[0].max(dim=1)[1]
            else:
                preds = logits.max(dim=1)[1]
            count += batch_size
            train_loss += loss.item() * batch_size
            train_true.append(label.cpu().numpy())
            train_pred.append(preds.detach().cpu().numpy())
            if (i + 1) % print_freq == 0:
                log_string(f"EPOCH {epoch:03d}/{args.epochs:03d} Batch {i:05d}/{len(train_loader):05d}: Loss {train_loss/count:.8f}")

        scheduler.step()
        train_true = np.concatenate(train_true)
        train_pred = np.concatenate(train_pred)
        train_loss = train_loss / count
        train_acc = metrics.accuracy_score(train_true, train_pred)
        train_avg_acc = metrics.balanced_accuracy_score(train_true, train_pred)
        log_string(f"TRAIN: loss {train_loss:.6f}, acc {train_acc:.6f}, avg acc {train_avg_acc:.6f}")

        is_best = False
        test_acc, test_avg_acc, test_loss = test(model, test_loader, criterion, device)

        if test_acc >= best_test_acc:
            best_test_acc = test_acc
            is_best = True
        saveID = utils.save_checkpoint({
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'best_test_acc': best_test_acc,
            }, epoch, args.save_dir, is_best, saveID)

        epoch_string(f"EPOCH {epoch:03d}/{args.epochs:03d} | Test: loss {test_loss:.6f}, acc {test_acc:.6f}, avg acc {test_avg_acc:.6f} | Train: loss {train_loss:.6f}, acc {train_acc:.6f}, avg acc {train_avg_acc:.6f} | lr {lr:.8f} | {time.strftime('%Y-%m-%d-%H-%M-%S')}")

        if vessl_on:
            base_name = f'pntcls/'
            log_dict = {'loss':test_loss, "acc":test_acc, "avg_acc":test_avg_acc, "best_acc":best_test_acc}
            log_dict = {base_name+k: log_dict[k] for k in log_dict}
            vessl.log(step=epoch, payload=log_dict)


def test(model, test_loader, criterion, device):
    test_loss = 0.0
    count = 0.0
    model.eval()
    test_pred = []
    test_true = []
    for data, label in test_loader:

        trot = None
        if args.rot_test == 'z':
            zang = torch.rand(data.shape[0]) * 2*np.pi
            trot = rmutil.aa2q(torch.stack([torch.zeros_like(zang), torch.zeros_like(zang), zang], -1)).to(device)
        elif args.rot_test == 'so3':
            trot = rmutil.qrand((data.shape[0],)).to(device)
        
        data, label = data.to(device), label.to(device, dtype=torch.long).squeeze()
        if trot is not None:
            data = rmutil.qaction(trot[...,None,:], data)
        data = data.permute(0, 2, 1)
        batch_size = data.size()[0]
        with torch.no_grad():
            logits = model(data)
            loss = criterion(logits, label)
            if args.model in ['original', 'bipointnet', 'evnet']:
                preds = logits[0].max(dim=1)[1]
            else:
                preds = logits.max(dim=1)[1]
        count += batch_size
        test_loss += loss.item() * batch_size
        test_true.append(label.cpu().numpy())
        test_pred.append(preds.detach().cpu().numpy())
    test_true = np.concatenate(test_true)
    test_pred = np.concatenate(test_pred)
    test_loss = test_loss / count
    test_acc = metrics.accuracy_score(test_true, test_pred)
    avg_per_class_acc = metrics.balanced_accuracy_score(test_true, test_pred)
    log_string(f"TEST: loss {test_loss:.6f}, acc {test_acc:.6f}, avg acc {avg_per_class_acc:.6f}")
    return test_acc, avg_per_class_acc, test_loss

def test2(model, loader, criterion, device):
    num_class=40
    mean_correct = []
    class_acc = np.zeros((num_class,3))
    for j, data in enumerate(loader):
        points, target = data
        
        trot = None
        if args.rot == 'z':
            zang = torch.rand(points.shape[0]) * 2*np.pi
            trot = rmutil.aa2q(torch.stack([torch.zeros_like(zang), torch.zeros_like(zang), zang], -1)).to(device)
        elif args.rot == 'so3':
            trot = rmutil.qrand((points.shape[0],)).to(device)
        if trot is not None:
            points = rmutil.qaction(trot[...,None,:], points)
        
        target = target[:, 0]
        points = points.transpose(2, 1)
        points, target = points.cuda(), target.cuda()
        classifier = model.eval()
        pred = classifier(points)
        if args.model in ['original', 'bipointnet']:
            pred = pred[0]
        pred_choice = pred.data.max(1)[1]
        for cat in np.unique(target.cpu()):
            classacc = pred_choice[target==cat].eq(target[target==cat].long().data).cpu().sum()
            class_acc[cat,0]+= classacc.item()/float(points[target==cat].size()[0])
            class_acc[cat,1]+=1
        correct = pred_choice.eq(target.long().data).cpu().sum()
        mean_correct.append(correct.item()/float(points.size()[0]))
    class_acc[:,2] =  class_acc[:,0]/ class_acc[:,1]
    class_acc = np.mean(class_acc[:,2])
    instance_acc = np.mean(mean_correct)
    log_string(f"TEST: acc {instance_acc:.6f}, avg acc {class_acc:.6f}")
    return instance_acc, class_acc, 0


if __name__ == "__main__":
    main()