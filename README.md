**Seg2linkUnet2d** is a package for training 2D U-Net, 
and using the trained 2D U-Net to make semantic segmentation 
in 3D bio-medical images.

Seg2linkUnet2d is based on PyTorch.

## Install
- Install [Anaconda](https://www.anaconda.com/products/individual) 
  or [Miniconda](https://conda.io/miniconda.html)
- Create a new conda environment and activate it by:
```console
$ conda create -n unet2-env python=3.9 pip
$ conda activate unet2-env
```
- Install [PyTorch](https://pytorch.org/get-started/locally/)
- Install Jupyter notebook
```console
$ pip install notebook
```
- Install seg2link:
```console
$ pip install seg2link
```

## Use the notebook for training/segmentation
- Activate the created environment by:
```console
$ conda activate unet2-env
```
- Start the jupyter notebook
```console
$ jupyter notebook
```
- Open the notebook under the /Example folder and run
the codes according to the instructions.
