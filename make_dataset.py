
import torchvision.transforms as trn
import torchvision.datasets as dset

import svhn_loader as svhn


# *** update this before running on your machine ***
# cifar10_path = '/nobackup/my_xfdu/cifarpy'
# cifar100_path = '/nobackup/my_xfdu/cifar-100-python'
# svhn_path = '/nobackup/my_xfdu/svhn/'
# lsun_c_path = '/nobackup/my_xfdu/LSUN_C'
# lsun_r_path = '/nobackup/my_xfdu/LSUN_resize'
# isun_path = '/nobackup/my_xfdu/iSUN'
# dtd_path = '/nobackup/my_xfdu/dtd/images'
# places_path = '/nobackup/my_xfdu/places365/'
# tinyimages_300k_path = '/nobackup/my_xfdu/300K_random_images.npy'


cifar10_path = '/nobackup-slow/dataset/my_xfdu/cifarpy/'
cifar100_path = '/nobackup-slow/dataset/my_xfdu/cifarpy/'
svhn_path = '/nobackup-slow/dataset/svhn/'
lsun_c_path = '/nobackup-slow/dataset/LSUN_C'
lsun_r_path = '/nobackup-slow/dataset/LSUN_resize'
isun_path = '/nobackup-slow/dataset/iSUN'
dtd_path = '/nobackup-slow/dataset/dtd/images'
places_path = '/nobackup-slow/dataset/my_xfdu/places365/'
tinyimages_300k_path = '/nobackup-slow/dataset/my_xfdu/300K_random_images.npy'





def load_CIFAR(dataset, classes=[]):

    mean = [x / 255 for x in [125.3, 123.0, 113.9]]
    std = [x / 255 for x in [63.0, 62.1, 66.7]]

    # train_transform = trn.Compose([trn.RandomHorizontalFlip(), trn.RandomCrop(32, padding=4),
    #                                trn.ToTensor(), trn.Normalize(mean, std)])
    train_transform = trn.Compose([trn.ToTensor(), trn.Normalize(mean, std)])
    test_transform = trn.Compose([trn.ToTensor(), trn.Normalize(mean, std)])

    if dataset in ['cifar10']:
        print('loading CIFAR-10')
        train_data = dset.CIFAR10(
            cifar10_path, train=True, transform=train_transform, download=True)
        test_data = dset.CIFAR10(
            cifar10_path, train=False, transform=test_transform, download=True)

    elif dataset in ['cifar100']:
        print('loading CIFAR-100')
        train_data = dset.CIFAR100(
            cifar100_path, train=True, transform=train_transform, download=True)
        test_data = dset.CIFAR100(
            cifar100_path, train=False, transform=test_transform, download=True)

    return train_data, test_data


def load_SVHN(transform, include_extra=False):


    print('loading SVHN')
    if not include_extra:
        train_data = svhn.SVHN(root=svhn_path, split="train",
                                 transform=transform)
    else:
        train_data = svhn.SVHN(root=svhn_path, split="train_and_extra",
                               transform=transform)

    test_data = svhn.SVHN(root=svhn_path, split="test",
                              transform=transform)

    train_data.targets = train_data.targets.astype('int64')
    test_data.targets = test_data.targets.astype('int64')
    return train_data, test_data


def load_ood_dataset(dataset, transform):


    if dataset == 'lsun_c':
        print('loading LSUN_C')
        out_data = dset.ImageFolder(root=lsun_c_path,
                                    transform=transform)

    if dataset == 'lsun_r':
        print('loading LSUN_R')
        out_data = dset.ImageFolder(root=lsun_r_path,
                                    transform=transform)

    if dataset == 'isun':
        print('loading iSUN')
        out_data = dset.ImageFolder(root=isun_path,
                                    transform=transform)
    if dataset == 'dtd':
        print('loading DTD')
        out_data = dset.ImageFolder(root=dtd_path,
                                    transform=transform)
    if dataset == 'places':
        print('loading Places365')
        out_data = dset.ImageFolder(root=places_path,
                                    transform=transform)
        import numpy as np
        import torch
        idx = np.array(range(len(out_data)))
        rng = np.random.choice(idx, 10000)
        idx = idx[rng]
        out_data = torch.utils.data.Subset(out_data, idx)

    return out_data


