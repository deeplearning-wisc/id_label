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
parser.add_argument('--load_ckpt', default='', type=str)
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


set_deterministic(args.seed)


if args.dataset.name == 'cifar10':
    import torchvision.datasets as dset

    train_data_in = dset.CIFAR10('/nobackup-slow/dataset/my_xfdu/cifarpy', train=True,
                                 transform=get_aug(train=False, train_classifier=False, **args.aug_kwargs))
    test_data = dset.CIFAR10('/nobackup-slow/dataset/my_xfdu/cifarpy', train=False,
                             transform=get_aug(train=False, train_classifier=False, **args.aug_kwargs))
    outlier_data = datasets.RandomImages(root=tinyimages_300k_path, transform=get_aug(train=False, train_classifier=False, **args.aug_kwargs))
    args.num_classes = 10

elif args.dataset.name == 'cifar100':

    class_list = None

    import torchvision.datasets as dset

    train_data_in = dset.CIFAR100('/nobackup-slow/dataset/my_xfdu/cifarpy/', train=True,
                                  transform=get_aug(train=False, train_classifier=False, **args.aug_kwargs))
    test_data = dset.CIFAR100('/nobackup-slow/dataset/my_xfdu/cifarpy/', train=False,
                              transform=get_aug(train=False, train_classifier=False, **args.aug_kwargs))
    outlier_data = datasets.RandomImages(root=tinyimages_300k_path,
                                         transform=get_aug(train=False, train_classifier=False, **args.aug_kwargs))
    args.num_classes = 100


train_loader = torch.utils.data.DataLoader(train_data_in,
                                           batch_size=args.train.batch_size,
                                           shuffle=True, num_workers=4)
outlier_data_loader = torch.utils.data.DataLoader(outlier_data,
                                           batch_size=args.train.batch_size,
                                           shuffle=True, num_workers=4)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.train.batch_size, shuffle=False, num_workers=2)

# define model
model = get_model(args.model, args).to(args.device)

model.load_state_dict(torch.load(
                    args.load_ckpt)['state_dict'])
model.eval()

def feat_extract(loader, ood=0, layer='penul'):
    targets = np.array([])
    features = []
    for idx, (x, labels) in enumerate(loader):
        # feat = model.backbone.features(x.to(device, non_blocking=True))
        ret_dict = model.forward_eval(x.to(args.device, non_blocking=True), layer=layer)
        feat = ret_dict['features']
        # breakpoint()
        if not ood:
            targets = np.append(targets, np.ones(len(feat)))
        else:
            targets = np.append(targets, np.zeros(len(feat)))
        features.append(feat.data.cpu().numpy())
    return np.concatenate(features), targets.astype(int)


ood = ['svhn', 'lsun_c', 'lsun_r', 'isun', 'places', 'dtd']
from make_dataset import load_ood_dataset, load_SVHN
def ood_feat_loader(name):
    if name == 'svhn':
        _, aux_out_data = load_SVHN(get_aug(train=False, train_classifier=False, **args.aug_kwargs))
    else:
        aux_out_data = load_ood_dataset(name, get_aug(train=False, train_classifier=False, **args.aug_kwargs))

    aux_out_data_loader = torch.utils.data.DataLoader(aux_out_data, batch_size=args.train.batch_size, shuffle=False, num_workers=2)
    features_test_n, ltest_n = feat_extract(aux_out_data_loader, ood=1, layer=args.layer)
    return features_test_n, ltest_n





features_test_in, ltest_in = feat_extract(test_loader, ood=0, layer=args.layer)
features_train_in, ltrain_in = feat_extract(train_loader, ood=0, layer=args.layer)
features_test_out, ltest_out = ood_feat_loader('svhn')
features_train_out, ltrain_out = feat_extract(outlier_data_loader, ood=1, layer=args.layer)



normalizer = lambda x: x / (np.linalg.norm(x, ord=2, axis=-1, keepdims=True) + 1e-10)
ftrain_in = normalizer(features_train_in)
ftrain_out = normalizer(features_train_out)
ftest_in = normalizer(features_test_in)
ftest_out = normalizer(features_test_out)

#######################  Linear Probe #######################

lp_acc, (clf_known, _, _, lp_preds_k), losses_train = get_linear_acc(np.concatenate([ftrain_in, ftrain_out],0),
                                                        np.concatenate([ltrain_in, ltrain_out],0),
                                                    np.concatenate([ftest_in, ftest_out],0),
                                                        np.concatenate([ltest_in, ltest_out],0),
                                                        2, epochs=50,
                                                       print_ret=True)
clf_known.eval()
out = clf_known(torch.from_numpy(np.concatenate([ftest_in, ftest_out],0)).cuda())
labels = torch.from_numpy(np.concatenate([np.ones(len(ftest_in)), np.zeros(len(ftest_out))],0)).cuda()
pred_scores = torch.sigmoid(out)
loss = F.binary_cross_entropy_with_logits(out.view(-1), labels.float())
print('loss:',loss)
from metric_utils import get_measures, print_measures
measures = get_measures(pred_scores[:len(ftest_in)].cpu().detach().numpy(), pred_scores[len(ftest_in):].cpu().detach().numpy(), plot=False)
print_measures(measures[0], measures[1], measures[2], 'svhn')

# breakpoint()
features_test_out, ltest_out = ood_feat_loader('dtd')
ftest_out = normalizer(features_test_out)
out = clf_known(torch.from_numpy(np.concatenate([ftest_in, ftest_out],0)).cuda())
labels = torch.from_numpy(np.concatenate([np.ones(len(ftest_in)), np.zeros(len(ftest_out))],0)).cuda()
pred_scores = torch.sigmoid(out)
loss = F.binary_cross_entropy_with_logits(out.view(-1), labels.float())
print('loss:',loss)
measures = get_measures(pred_scores[:len(ftest_in)].cpu().detach().numpy(), pred_scores[len(ftest_in):].cpu().detach().numpy(), plot=False)
print_measures(measures[0], measures[1], measures[2], 'dtd')



features_test_out, ltest_out = ood_feat_loader('lsun_c')
ftest_out = normalizer(features_test_out)
out = clf_known(torch.from_numpy(np.concatenate([ftest_in, ftest_out],0)).cuda())
labels = torch.from_numpy(np.concatenate([np.ones(len(ftest_in)), np.zeros(len(ftest_out))],0)).cuda()
pred_scores = torch.sigmoid(out)
loss = F.binary_cross_entropy_with_logits(out.view(-1), labels.float())
print('loss:',loss)
measures = get_measures(pred_scores[:len(ftest_in)].cpu().detach().numpy(), pred_scores[len(ftest_in):].cpu().detach().numpy(), plot=False)
print_measures(measures[0], measures[1], measures[2], 'lsun_c')



features_test_out, ltest_out = ood_feat_loader('lsun_r')
ftest_out = normalizer(features_test_out)
out = clf_known(torch.from_numpy(np.concatenate([ftest_in, ftest_out],0)).cuda())
labels = torch.from_numpy(np.concatenate([np.ones(len(ftest_in)), np.zeros(len(ftest_out))],0)).cuda()
pred_scores = torch.sigmoid(out)
loss = F.binary_cross_entropy_with_logits(out.view(-1), labels.float())
print('loss:',loss)
measures = get_measures(pred_scores[:len(ftest_in)].cpu().detach().numpy(), pred_scores[len(ftest_in):].cpu().detach().numpy(), plot=False)
print_measures(measures[0], measures[1], measures[2], 'lsun_r')



features_test_out, ltest_out = ood_feat_loader('isun')
ftest_out = normalizer(features_test_out)
out = clf_known(torch.from_numpy(np.concatenate([ftest_in, ftest_out],0)).cuda())
labels = torch.from_numpy(np.concatenate([np.ones(len(ftest_in)), np.zeros(len(ftest_out))],0)).cuda()
pred_scores = torch.sigmoid(out)
loss = F.binary_cross_entropy_with_logits(out.view(-1), labels.float())
print('loss:',loss)
measures = get_measures(pred_scores[:len(ftest_in)].cpu().detach().numpy(), pred_scores[len(ftest_in):].cpu().detach().numpy(), plot=False)
print_measures(measures[0], measures[1], measures[2], 'isun')



features_test_out, ltest_out = ood_feat_loader('places')
ftest_out = normalizer(features_test_out)
out = clf_known(torch.from_numpy(np.concatenate([ftest_in, ftest_out],0)).cuda())
labels = torch.from_numpy(np.concatenate([np.ones(len(ftest_in)), np.zeros(len(ftest_out))],0)).cuda()
pred_scores = torch.sigmoid(out)
loss = F.binary_cross_entropy_with_logits(out.view(-1), labels.float())
print('loss:',loss)
measures = get_measures(pred_scores[:len(ftest_in)].cpu().detach().numpy(), pred_scores[len(ftest_in):].cpu().detach().numpy(), plot=False)
print_measures(measures[0], measures[1], measures[2], 'places')

if args.dataset.name == 'cifar10':
    test_data_out = dset.CIFAR100('/nobackup-slow/dataset/my_xfdu/cifarpy/', train=False,
                              transform=get_aug(train=False, train_classifier=False, **args.aug_kwargs))
    test_data_out_loader = torch.utils.data.DataLoader(test_data_out, batch_size=args.train.batch_size, shuffle=False,
                                                      num_workers=2)
    features_test_out, ltest_out = feat_extract(test_data_out_loader, ood=1, layer=args.layer)
    ftest_out = normalizer(features_test_out)

    out = clf_known(torch.from_numpy(np.concatenate([ftest_in, ftest_out], 0)).cuda())
    labels = torch.from_numpy(np.concatenate([np.ones(len(ftest_in)), np.zeros(len(ftest_out))],0)).cuda()
    pred_scores = torch.sigmoid(out)
    loss = F.binary_cross_entropy_with_logits(out.view(-1), labels.float())
    print('loss:', loss)


    measures = get_measures(pred_scores[:len(ftest_in)].cpu().detach().numpy(), pred_scores[len(ftest_in):].cpu().detach().numpy(), plot=False)
    print_measures(measures[0], measures[1], measures[2], 'c100')
else:
    test_data_out = dset.CIFAR10('/nobackup-slow/dataset/my_xfdu/cifarpy/', train=False,
                                  transform=get_aug(train=False, train_classifier=False, **args.aug_kwargs))
    test_data_out_loader = torch.utils.data.DataLoader(test_data_out, batch_size=args.train.batch_size, shuffle=False,
                                                       num_workers=2)
    features_test_out, ltest_out = feat_extract(test_data_out_loader, ood=1, layer=args.layer)
    ftest_out = normalizer(features_test_out)

    out = clf_known(torch.from_numpy(np.concatenate([ftest_in, ftest_out], 0)).cuda())
    labels = torch.from_numpy(np.concatenate([np.ones(len(ftest_in)), np.zeros(len(ftest_out))],0)).cuda()
    pred_scores = torch.sigmoid(out)
    loss = F.binary_cross_entropy_with_logits(out.view(-1), labels.float())
    print('loss:', loss)

    measures = get_measures(pred_scores[:len(ftest_in)].cpu().detach().numpy(),
                            pred_scores[len(ftest_in):].cpu().detach().numpy(), plot=False)
    print_measures(measures[0], measures[1], measures[2], 'c10')