# ID Label

This is the source code accompanying the paper [When and How Does In-distribution Label Help Out-of-Distribution Detection?](https://arxiv.org/abs/2405.18635) by Xuefeng Du, Yiyou Sun, and Yixuan Li


## Ads 

Check out our ICLR'24 [SAL](https://github.com/deeplearning-wisc/sal) on analyzing the effect of the unlabeled data for OOD detection if you are interested!



## Dataset Preparation


**CIFAR-10/CIFAR-100**

* The dataloader will download it automatically when first running the programs.

**OOD datasets**


* The OOD datasets with CIFAR-100 as in-distribution are 5 OOD datasets, i.e., SVHN, PLACES365, LSUN-C, LSUN-R, TEXTURES.
* Please refer to Part 1 and 2 of the codebase [here](https://github.com/deeplearning-wisc/knn-ood). 

## Training

Please execute the following in the command shell for the unsupervised case (cifar100 as ID):
```
python run.py --config-file configs/my_resnet_mlp1000_norelu_cifar100.yaml --add none --gamma_u 1 --gamma_l 1
```
and 
```
python run.py --config-file configs/my_resnet_mlp1000_norelu_cifar100.yaml --add combine --gamma_u 3 --gamma_l 0.0225
```
for the supervised case.

Please execute the following in the command shell for the unsupervised case (cifar10 as ID):
```
python run.py --config-file configs/my_resnet_mlp1000_norelu_cifar10.yaml --add none --gamma_u 1 --gamma_l 1
```
and 
```
python run.py --config-file configs/my_resnet_mlp1000_norelu_cifar10.yaml --add combine --gamma_u 0.5 --gamma_l 0.25
```
for the supervised case.

## Linear Probing

To run linear probing when the test ood distribution is the same as the training outliers, run:
```
python lp_same_dis.py --load_ckpt c100_sup.pth --ood_name svhn --config-file configs/my_resnet_mlp1000_norelu_cifar100.yaml
```
"ood_name" denotes the type of OOD data, and "load_ckpt" denotes the pretrained model.

To run linear probing when the test ood distribution is different from the training outliers, run:
```
python lp.py --load_ckpt c100_sup.pth --config-file configs/my_resnet_mlp1000_norelu_cifar100.yaml
```




### Citing

If you find our code useful, please consider citing:

```
@inproceedings{du2024when,
      title={When and How Does In-Distribution Label Help Out-of-Distribution Detection?}, 
      author={Xuefeng Du and Yiyou Sun and Yixuan Li},
      booktitle = {International Conference on Machine Learning},
      year = {2024}
}
```
