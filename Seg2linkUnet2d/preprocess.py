import itertools
import os
from pathlib import Path
from typing import List, Tuple

import torch
from PIL import Image
import numpy as np
from numpy import ndarray
from torch import nn


def _make_folder(path_i, print_=True):
    """
    Make a folder
    Parameters
    ----------
    path_i : str
         The folder path
    print_ : bool, optional
        If True, print the relative path of the created folder. Default: True
    Returns
    -------
    path_i : str
        The folder path
    """
    if not os.path.exists(path_i):
        os.makedirs(path_i)
    if print_:
        print(os.path.relpath(path_i, os.getcwd()))
    return path_i


def load_image(path: str, print_: bool = True) -> List[ndarray]:
    """Load images as list into RAM"""
    paths_list = get_files(Path(path))
    if not paths_list:
        raise ValueError(f"No tif files found in the path: {path}")

    img_list = []
    for z, img_path in enumerate(paths_list):
        img_list.append(np.array(Image.open(img_path)))

    if print_:
        print(f"Load {len(img_list)} images. Shape (image #1): {img_list[0].shape}")

    return img_list


def load_filenames(path: str) -> List[str]:
    """Load images as list into RAM"""
    paths_list = get_files(Path(path))
    if not paths_list:
        raise ValueError(f"No tif files found in the path: {path}")
    return paths_list


def load_one_image(path: str):
    return np.array(Image.open(path))


def get_files(path: Path) -> List[str]:
    """Return all paths of .tiff or .tif files"""
    return [str(file) for file in sorted(path.glob("*.tif*"))]


class Conv3dPytorch:
    def __init__(self, filter_size):
        self.conv3 = nn.Conv3d(1, 1, filter_size, padding='same', bias=False)
        self.conv3.weight = nn.Parameter(torch.ones_like(self.conv3.weight))

    def predict(self, img3d):
        x = torch.unsqueeze(torch.unsqueeze(torch.Tensor(img3d), 0), 0)
        return torch.squeeze(self.conv3(x)).detach().numpy()


def lcn_gpu(img3d, noise_level=5, filter_size=(27, 27, 1)):
    """
    Local contrast normalization by gpu
    Parameters
    ----------
    img3d : numpy.ndarray
        The raw 3D image
    noise_level : float
        The parameter to suppress the enhancement of the background noises
    filter_size : tuple, optional
        the window size to apply the normalization along x, y, and z axis. Default: (27, 27, 1)
    Returns
    -------
    norm : numpy.ndarray
        The normalized 3D image
    Notes
    -----
    The normalization in the edge regions currently used zero padding based on torch.nn.Conv3D function,
    which is different with the lcn_cpu function (uses "reflect" padding).
    """
    volume = filter_size[0] * filter_size[1] * filter_size[2]
    conv3d_model = Conv3dPytorch(filter_size)
    avg = conv3d_model.predict(img3d) / volume
    diff_sqr = np.square(img3d - avg)
    std = np.sqrt(conv3d_model.predict(diff_sqr) / volume)
    return np.divide(img3d - avg, std + noise_level)


def _normalize_image(images: List[ndarray]) -> Tuple[List[ndarray], float, float]:
    """
    Normalize 2D images by standardization
    """
    mean = np.mean([np.mean(img) for img in images])
    std = np.mean([np.std(img) for img in images])
    return [(img - mean) / std for img in images], mean, std


def _normalize_label(label_imgs: List[ndarray]) -> List[ndarray]:
    """
    Transform cell/non-cell image into binary (0/1)
    """
    return [label > 0 for label in label_imgs]


def divide_flip_rotate(imgs: List[ndarray], size_subimages: Tuple[int, int]) -> ndarray:
    divided_images = divide_imgs(imgs, size_subimages)
    print(f"Divide into {divided_images.shape[0]} images")
    flipped_images = horizontal_flip(divided_images)
    print(f"Flipped into {flipped_images.shape[0]} images")
    rotated_images = rotate_imgs(flipped_images, axes=(1, 2))
    print(f"Rotated into {rotated_images.shape[0]} images")
    return np.expand_dims(rotated_images, axis=1)


def horizontal_flip(imgs: ndarray):
    return np.concatenate((imgs, np.fliplr(imgs)), axis=0)


def rotate_imgs(imgs: ndarray, axes: tuple):
    rotated_q1 = np.rot90(imgs, axes=axes)
    rotated_q2 = np.rot90(rotated_q1, axes=axes)
    rotated_q3 = np.rot90(rotated_q2, axes=axes)
    return np.concatenate((imgs, rotated_q1, rotated_q2, rotated_q3), axis=0)


def divide_imgs(imgs: List[ndarray], size_subimages: Tuple[int, int]) -> ndarray:
    subimgs = []
    for img in imgs:
        subimgs.extend(divide_img(img, size_subimages))
    return np.array(subimgs)


def divide_img(img: ndarray, size_subimages: Tuple[int, int]) -> List[ndarray]:
    x_siz, y_siz = img.shape
    x_input, y_input = size_subimages
    img_list = []
    for i, j in itertools.product(range(x_siz * 2 // x_input),
                                     range(y_siz * 2 // y_input)):
        idx_x = i * x_input // 2 if i * x_input // 2 + x_input <= x_siz else x_siz - x_input
        idx_y = j * y_input // 2 if j * y_input // 2 + y_input <= y_siz else y_siz - y_input
        img_list.append(img[idx_x:idx_x + x_input, idx_y:idx_y + y_input])
    return img_list