
[![PyPI](https://img.shields.io/pypi/v/seg2link-unet2d)](https://pypi.org/project/seg2link-unet2d/) [![GitHub](https://img.shields.io/github/license/WenChentao/3DeeCellTracker)](https://github.com/WenChentao/3DeeCellTracker/blob/master/LICENSE)

**Seg2linkUnet2d** is a package for training a 2D U-Net to predict cell/non-cell regions 
in 3D bio-medical images.

Seg2linkUnet2d and [Seg2Link](https://github.com/WenChentao/Seg2Link) can be used together to perform semi-automatic 3D cell segmentation. 
Read the [documentation](https://wenchentao.github.io/Seg2Link/seg2link-unet2d.html) to learn how to do it.


## Install
- Install [Anaconda](https://www.anaconda.com/products/individual) 
  or [Miniconda](https://conda.io/miniconda.html)
- Create a new conda environment and activate it by:
```console
$ conda create -n unet2
$ conda activate unet2
```
- Install [PyTorch](https://pytorch.org/get-started/locally/)
- Install seg2link_unet2d:
```console
$ pip install seg2link-unet2d[local]
```

## Use the notebook for training/segmentation
- Activate the created environment by:
```console
$ conda activate unet2
```
- Start the jupyter notebook
```console
$ jupyter notebook
```
- Open the notebook under the /Examples folder and run
the codes according to the instructions.
