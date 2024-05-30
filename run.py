import time
import os
import torch
import torch.nn as nn
import torch.nn.functional as F 
import torchvision
from itertools import cycle
import numpy as np
import argparse
from arguments import set_deterministic, Namespace, csv, shutil, yaml
from augmentations import get_aug
from models import get_model
from optimizers import get_optimizer, LR_Scheduler
from datetime import date
from sklearn.cluster import KMeans
from ylib.ytool import cluster_acc
import open_world_cifar as datasets
from linear_probe import get_linear_acc

tinyimages_300k_path = '/nobackup-slow/dataset/my_xfdu/300K_random_images.npy'

def main(log_writer, log_file, device, args):
    iter_count = 0

    dataroot = args.data_dir
    if args.dataset.name == 'cifar10':
        import torchvision.datasets as dset
        train_data_in = dset.CIFAR10('/nobackup-slow/dataset/my_xfdu/cifarpy', train=True, transform=get_aug(train=True, **args.aug_kwargs))
        args.num_classes = 10
    elif args.dataset.name == 'cifar100':
        import torchvision.datasets as dset
        train_data_in = dset.CIFAR100('/nobackup-slow/dataset/my_xfdu/cifarpy', train=True,
                                     transform=get_aug(train=True, **args.aug_kwargs))
        args.num_classes = 100

    train_loader = torch.utils.data.DataLoader(train_data_in,
                                               batch_size=args.train.batch_size,
                                               shuffle=True, num_workers=4, drop_last=True)

    # define model
    model = get_model(args.model, args).to(device)

    # define optimizer
    optimizer = get_optimizer(
        args.train.optimizer.name, model, 
        lr=args.train.base_lr*args.train.batch_size/256, 
        momentum=args.train.optimizer.momentum,
        weight_decay=args.train.optimizer.weight_decay)

    lr_scheduler = LR_Scheduler(
        optimizer,
        args.train.warmup_epochs, args.train.warmup_lr*args.train.batch_size/256, 
        args.train.num_epochs, args.train.base_lr*args.train.batch_size/256, args.train.final_lr*args.train.batch_size/256, 
        len(train_loader),
        constant_predictor_lr=True
    )

    ckpt_dir = os.path.join(args.log_dir, "checkpoints")
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)

    for epoch in range(0, args.train.stop_at_epoch):

        #######################  Train #######################
        model.train()
        print("number of iters this epoch: {}".format(len(train_loader)))
        # unlabel_loader_iter = cycle(train_unlabel_loader)
        for idx, ((x1, x2, ux1, ux2), target) in enumerate(train_loader):
            # ((ux1, ux2), target_unlabeled) = next(unlabel_loader_iter)
            # breakpoint()
            # x1, x2, target = x1.to(device), x2.to(device), target.to(device)
            x1, x2, ux1, ux2, target = x1.to(device), x2.to(device), ux1.to(device), ux2.to(device), target.to(device)

            model.zero_grad()
            data_dict = model.forward_my(x1, x2, ux1, ux2, target, add=args.add)
            loss = data_dict['loss'].mean()
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            data_dict.update({'lr':lr_scheduler.get_lr()})
            if (idx + 1) % args.print_freq == 0:
                print('Lr: ', lr_scheduler.get_lr())
                loss1, loss2, loss3, loss4, loss5 = data_dict["d_dict"]["loss1"].item(), data_dict["d_dict"][
                    "loss2"].item(), data_dict["d_dict"]["loss3"].item(), data_dict["d_dict"]["loss4"].item(), \
                                                    data_dict["d_dict"]["loss5"].item()

                print(
                    'Train: [{0}][{1}/{2}]\t Loss_all {3:.3f} \tc1:{4:.2e}\tc2:{5:.3f}\tc3:{6:.2e}\tc4:{7:.2e}\tc5:{8:.3f}'.format(
                        epoch, idx + 1, len(train_loader), loss.item(), loss1, loss2, loss3, loss4, loss5
                    ))


        #######################  Evaluation #######################
        model.eval()


        #######################  Save Epoch #######################
        if (epoch + 1) % args.log_freq == 0:
            model_path = os.path.join(ckpt_dir, f"{epoch + 1}.pth")
            torch.save({
                'epoch': epoch + 1,
                'state_dict': model.state_dict()
            }, model_path)
            print(f"Model saved to {model_path}")

    # Save checkpoint
    model_path = os.path.join(ckpt_dir, f"latest_{epoch+1}.pth")
    torch.save({
        'epoch': epoch+1,
        'state_dict':model.state_dict()
    }, model_path)
    print(f"Model saved to {model_path}")
    with open(os.path.join(args.log_dir, "checkpoints", f"checkpoint_path.txt"), 'w+') as f:
        f.write(f'{model_path}')


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config-file', default='configs/supspectral_resnet_mlp1000_norelu_cifar100.yaml', type=str)
    # parser.add_argument('-c', '--config-file', default='configs/spectral_resnet_mlp1000_norelu_cifar10_lr003_mu1.yaml', type=str)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--log_freq', type=int, default=100)
    parser.add_argument('--workers', type=int, default=32)
    parser.add_argument('--test_bs', type=int, default=80)
    parser.add_argument('--download', action='store_true', help="if can't find dataset, download from web")
    parser.add_argument('--data_dir', type=str, default='/nobackup-slow/dataset/my_xfdu/cifarpy')
    parser.add_argument('--dist_url', type=str, default='tcp://localhost:10001')
    parser.add_argument('--log_dir', type=str, default='/nobackup-fast/dataset/my_xfdu/add_label/logs/')
    parser.add_argument('--ckpt_dir', type=str, default='/nobackup-fast/dataset/my_xfdu/add_label/')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--eval_from', type=str, default=None)
    parser.add_argument('--hide_progress', action='store_true')
    parser.add_argument('--vis_freq', type=int, default=2000)
    parser.add_argument('--deep_eval_freq', type=int, default=50)
    parser.add_argument('--print_freq', type=int, default=10)
    parser.add_argument('--labeled-num', default=80, type=int)
    parser.add_argument('--labeled-ratio', default=1, type=float)
    parser.add_argument('--gamma_l', default=0.0225, type=float)
    parser.add_argument('--gamma_u', default=3, type=float)
    # parser.add_argument('--gamma_l', default=1, type=float)
    # parser.add_argument('--gamma_u', default=1, type=float)
    parser.add_argument('--c3_rate', default=1, type=float)
    parser.add_argument('--c4_rate', default=2, type=float)
    parser.add_argument('--c5_rate', default=1, type=float)
    parser.add_argument('--add', default='none', type=str)
    parser.add_argument('--proj_feat_dim', default=1000, type=int)
    parser.add_argument('--went', default=0.0, type=float)
    parser.add_argument('--momentum_proto', default=0.95, type=float)
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--base_lr', default=0.03, type=float)
    parser.add_argument('--layer', default='penul', type=str)

    args = parser.parse_args()

    with open(args.config_file, 'r') as f:
        for key, value in Namespace(yaml.load(f, Loader=yaml.FullLoader)).__dict__.items():
            if key not in vars(args):
                vars(args)[key] = value

    assert not None in [args.log_dir, args.data_dir, args.ckpt_dir, args.name]

    alpha = args.gamma_l
    beta = args.gamma_u
    scale = 1
    args.c1, args.c2 = 2 * alpha * scale, 2 * beta * scale
    args.c3, args.c4, args.c5 = alpha ** 2 * scale, \
                 alpha * beta * scale * 2, \
                 beta ** 2 * scale

    args.train.base_lr = args.base_lr

    disc = f"labelnum-{args.labeled_num}-c1-{args.c1:.2f}-c2-{args.c2:.1f}-c3-{args.c3:.1e}-c4-{args.c4:.1e}-c5-{args.c5:.1e}-gamma_l-{args.gamma_l:.2f}-gamma_u-{args.gamma_u:.2f}-r345-{args.c3_rate}-{args.c4_rate}-{args.c5_rate}"+ \
           f"-lr{args.base_lr}-layer{args.layer}-seed{args.seed}"
    args.log_dir = os.path.join(args.log_dir, 'add-' + args.add + '-'+'{}'.format(date.today())+args.name+'-{}'.format(disc))

    os.makedirs(args.log_dir, exist_ok=True)
    print(f'creating file {args.log_dir}')
    os.makedirs(args.ckpt_dir, exist_ok=True)

    shutil.copy2(args.config_file, args.log_dir)
    set_deterministic(args.seed)

    vars(args)['aug_kwargs'] = {
        'name': args.model.name,
        'image_size': args.dataset.image_size
    }
    vars(args)['dataset_kwargs'] = {
        'dataset':args.dataset.name,
        'data_dir': args.data_dir,
        'download':args.download,
    }
    vars(args)['dataloader_kwargs'] = {
        'drop_last': True,
        'pin_memory': True,
        'num_workers': args.dataset.num_workers,
    }

    log_file = open(os.path.join(args.log_dir, 'log.csv'), mode='w')
    fieldnames = ['epoch', 'lr', 'kmeans_acc_train', 'kmeans_acc_test', 'kmeans_overall_acc', 'lp_acc']
    log_writer = csv.DictWriter(log_file, fieldnames=fieldnames)
    log_writer.writeheader()

    return args, log_file, log_writer


if __name__ == "__main__":
    args, log_file, log_writer = get_args()

    main(log_writer, log_file, device=args.device, args=args)

    completed_log_dir = args.log_dir.replace('in-progress', 'debug' if args.debug else 'completed')

    os.rename(args.log_dir, completed_log_dir)
    print(f'Log file has been saved to {completed_log_dir}')