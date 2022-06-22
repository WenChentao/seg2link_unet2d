from typing import Tuple, List

import itertools
from datetime import datetime
import os
from collections import namedtuple
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from numpy import ndarray
import torch
from torch import nn, optim
from torch.utils.data import TensorDataset, DataLoader
from PIL import Image

from seg2link_unet2d.preprocess import load_image, _make_folder, _normalize_image, _normalize_label, divide_flip_rotate, \
    load_filenames, load_one_image

TITLE_STYLE = {'fontsize': 16, 'verticalalignment': 'bottom'}
SIZE_SUBIMAGES = (160, 160)

ImageSize = namedtuple("ImageSize", ["x", "y"])
DataPaths = namedtuple("DataPaths", ["train_image", "train_cells", "test_image", "test_cells", "models"])
DataRaw = namedtuple("DataRaw", ["train_image", "train_cells"])
DataNorm = namedtuple("DataNorm", ["train_image", "train_cells"])
Train_Stat = namedtuple("Train_Stat", ["mean", "std"])


class UNet2(nn.Module):
    def __init__(self):
        super().__init__()
        pool_size = up_size = (2, 2)
        self.conv_ = ConvTwice(1, 16)
        self.down1 = Down(16, 32, pool_size)
        self.down2 = Down(32, 64, pool_size)
        self.down3 = Down(64, 64, pool_size)
        self.up1 = Up(128, 32, up_size)
        self.up2 = Up(64, 16, up_size)
        self.up3 = Up(32, 8, up_size)
        self.predict = Predict(8)

    def forward(self, x):
        x0 = self.conv_(x)
        x1 = self.down1(x0)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x = self.up1(x3, x2)
        x = self.up2(x, x1)
        x = self.up3(x, x0)
        return self.predict(x)


class Predict(nn.Module):
    def __init__(self, channels_in: int):
        super().__init__()
        self.predict = nn.Sequential(
            nn.Conv2d(channels_in, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.predict(x)


class Down(nn.Module):
    def __init__(self, channels_in: int, channels_out: int, pool_size: Tuple[int, int]):
        super().__init__()
        self.down = nn.Sequential(
            nn.MaxPool2d(pool_size, ceil_mode=True),
            ConvTwice(channels_in, channels_out),
        )

    def forward(self, x):
        return self.down(x)


class Up(nn.Module):
    def __init__(self, channels_in: int, channels_out: int, up_size: Tuple[int, int]):
        super().__init__()
        self.up = nn.Upsample(scale_factor=up_size)
        self.concat = torch.cat
        self.conv_ = ConvTwice(channels_in, channels_out)

    def forward(self, x1, x2):
        x = self.up(x1)
        x2_shape = x2.shape
        x = self.concat((x[:, :, :x2_shape[2], :x2_shape[3]], x2), dim=1)
        return self.conv_(x)


class ConvTwice(nn.Module):
    def __init__(self, channels_in: int, channels_out: int):
        super().__init__()
        channels_mid = max(
            max(channels_in, channels_out) // 2,
            min(channels_in, channels_out)
        )
        self.conv_twice = nn.Sequential(
            nn.Conv2d(channels_in, channels_mid, 3, padding='same'),
            nn.BatchNorm2d(channels_mid),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(channels_mid, channels_out, 3, padding='same'),
            nn.BatchNorm2d(channels_out),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv_twice(x)


def unet2_prediction(img: ndarray, model: UNet2):
    return np.squeeze(model(torch.Tensor(np.expand_dims(img, axis=(0, 1)))).detach().numpy())


class TrainingUNet2D:
    """
    Class to train the 2D U-Net

    Attributes
    ----------
    data_path : str
        The folder to store the data and results
    model : nn.Module
        A 2D U-Net model (before compiling)
    train_img1_size: Tuple[int, int]
        Size of train image #1
    raw_data: tuple
        Raw data of train images, labels
    norm_data: tuple
        Normalized data of train images, labels
    paths : Tuple[str, str, str, str, str]
        paths of train images, labels, test images, labels and models
    """

    def __init__(self, data_path: str, model: nn.Module):
        self.test_image_filenames = []
        self.train_loader = None
        self.train_img1_size = ImageSize(0, 0)
        self.train_stat = Train_Stat(0, 0)
        self.data_path = data_path
        self.model = model()
        self.paths = DataPaths("", "", "", "", "")
        self.raw_data = DataRaw([], [])
        self.norm_data = DataNorm([], [])
        self.train_data = TensorDataset()
        self.train_acc = []
        self.current_epoch = 1
        self.optimizer = optim.Adam(self.model.parameters())
        self.make_folders()

    def make_folders(self):
        """
        make folders for storing data and results
        """
        print("Made folders under:", os.getcwd())
        data_path = self.data_path
        print("Following folders were made: ")
        train_image_path = _make_folder(os.path.join(data_path, "train_image/"))
        train_label_path = _make_folder(os.path.join(data_path, "train_label/"))
        test_image_path = _make_folder(os.path.join(data_path, "raw_image/"))
        test_label_path = _make_folder(os.path.join(data_path, "raw_label/"))
        models_path = _make_folder(os.path.join(data_path, f"models_{datetime.now().strftime('%Y_%h_%d-%H_%M_%S')}/"))
        self.paths = DataPaths(train_image_path, train_label_path, test_image_path, test_label_path, models_path)

    def load_dataset(self):
        """
        Load training dataset and validation dataset stored in the corresponding folders
        """
        train_image = load_image(self.paths.train_image)
        train_label = load_image(self.paths.train_cells)
        self.train_img1_size = ImageSize(*train_image[0].shape)
        self.raw_data = DataRaw(train_image, train_label)

    def draw_dataset(self, percentile_top=99.9, percentile_bottom=0.1):
        """
        Draw the training dataset and validation dataset by max projection
        Parameters
        ----------
        percentile_top : float, optional
            A percentile to indicate the upper limitation for showing the images. Default: 99.9
        percentile_bottom : float, optional
            A percentile to indicate the lower limitation for showing the images. Default: 10
        """
        axs = self._subplots_2images(percentile_bottom, percentile_top,
                                     self.raw_data.train_image, self.raw_data.train_cells)
        axs[0].set_title("Image #1 (train)", fontdict=TITLE_STYLE)
        axs[1].set_title("Cell annotation #1 (train)", fontdict=TITLE_STYLE)
        plt.tight_layout()
        plt.pause(0.1)

    def normalize(self):
        """
        Normalize the images and divided them into small images for training the model
        """
        train_image_norm, mean, std = _normalize_image(self.raw_data.train_image)
        self.train_stat = Train_Stat(mean, std)
        train_label_norm = _normalize_label(self.raw_data.train_cells)
        self.norm_data = DataNorm(train_image_norm, train_label_norm)
        print("Images were normalized")

    def divide_images(self, batch_size=16):
        train_subimage = divide_flip_rotate(self.norm_data.train_image, SIZE_SUBIMAGES).astype(np.float32)
        train_subcells = divide_flip_rotate(self.norm_data.train_cells, SIZE_SUBIMAGES)

        self.train_data = TensorDataset(torch.tensor(train_subimage),
                                        torch.tensor(train_subcells, dtype=torch.bool))
        self.train_loader = DataLoader(dataset=self.train_data, batch_size=batch_size, shuffle=True)
        print(f"Images were divided into {len(self.train_data)} sub-images with shape: {self.train_data[0][0].shape}")

    def draw_norm_dataset(self, percentile_top=99.9, percentile_bottom=0.1):
        """
        Draw the normalized training dataset and validation dataset by max projection
        Parameters
        ----------
        percentile_top : float, optional
            A percentile to indicate the upper limitation for showing the images. Default: 99.9
        percentile_bottom : float, optional
            A percentile to indicate the lower limitation for showing the images. Default: 10
        """
        axs = self._subplots_2images(percentile_bottom, percentile_top,
                                     self.norm_data.train_image, self.norm_data.train_cells)
        axs[0].set_title("Normalized image #1 (train)", fontdict=TITLE_STYLE)
        axs[1].set_title("Cell annotation #1 (train)", fontdict=TITLE_STYLE)
        plt.tight_layout()
        plt.pause(0.1)

    @staticmethod
    def _subplots_2images(percentile_bottom, percentile_top, imgs_1: List[ndarray], imgs_2: List[ndarray]):
        """Make a (2, 2) layout figure to show 4 images"""
        siz_x, siz_y = imgs_1[0].shape
        if siz_x > siz_y:
            imgs_1_ = imgs_1[0].transpose()
            imgs_2_ = imgs_2[0].transpose()
            siz_y, siz_x = siz_x, siz_y
        else:
            imgs_1_ = imgs_1[0]
            imgs_2_ = imgs_2[0]
        fig, axs = plt.subplots(1, 2, figsize=(20, int(12 * siz_x / siz_y)))
        vmax_train = np.percentile(imgs_1, percentile_top)
        vmin_train = np.percentile(imgs_1, percentile_bottom)
        axs[0].imshow(imgs_1_, vmin=vmin_train, vmax=vmax_train, cmap="gray")
        axs[1].imshow(imgs_2_, cmap="gray")
        return axs

    def draw_divided_train_data(self, percentile_top=99.9, percentile_bottom=0.1):
        """
        Draw the previous 16 divided small images and corresponding cell images in training dataset by max projection
        Parameters
        ----------
        percentile_top : float, optional
            A percentile to indicate the upper limitation for showing the images. Default: 99.9
        percentile_bottom : float, optional
            A percentile to indicate the lower limitation for showing the images. Default: 10
        """
        vmax_train = np.percentile(self.norm_data.train_image, percentile_top)
        vmin_train = np.percentile(self.norm_data.train_image, percentile_bottom)
        fig, axs = plt.subplots(4, 8, figsize=(20, 10))
        idx = np.random.randint(len(self.train_data), size=16)
        for i, j in itertools.product(range(4), range(4)):
            axs[i, 2 * j].imshow(self.train_data.tensors[0][idx[i * 4 + j], 0, :, :].numpy(),
                                 vmin=vmin_train, vmax=vmax_train, cmap="gray")
            axs[i, 2 * j].axis("off")
        for i, j in itertools.product(range(4), range(4)):
            axs[i, 2 * j + 1].imshow(self.train_data.tensors[1][idx[i * 4 + j], 0, :, :].numpy(),
                                     cmap="gray")
            axs[i, 2 * j + 1].axis("off")
        plt.tight_layout()
        plt.pause(0.1)

    def train(self, iteration=30, weights_name="weights_training_"):
        """
        Train the 2D U-Net model
        Parameters
        ----------
        iteration : int, optional
            The number of epochs to train the model. Default: 100
        weights_name : str, optional
            The prefix of the weights files to be stored during training.
        Notes
        -----
        The training can be stopped by pressing Ctrl + C if users feel the prediction is good enough during training.
        Every time the train loss was reduced, the weights file will be stored into the /models folder
        """
        loss_func = nn.BCELoss()
        epoch_length = 32

        if self.current_epoch == 1:
            train_accuracy, train_loss = self.predict_train(loss_func)
            print(f"(Before training) Train loss: {train_loss}, Train accuracy: {train_accuracy}")

        start_epoch = self.current_epoch
        end_epoch = self.current_epoch + iteration
        for epoch in range(start_epoch, end_epoch):
            train_loss = 0
            n = 0
            with tqdm(total=epoch_length, desc=f'Epoch {epoch}/{end_epoch - 1}', ncols=50, unit='batch') as pbar:
                for X, y in self.train_loader:
                    X_prediction = self.model(X)
                    X_loss = loss_func(X_prediction, y.to(torch.float))
                    train_loss += X_loss.item()
                    n += 1
                    self.optimizer.zero_grad()
                    X_loss.backward()
                    self.optimizer.step()
                    pbar.update(1)
                    pbar.set_postfix(**{'Train loss': train_loss / n})
                    if n > epoch_length:
                        break

            train_accuracy, train_loss = self.predict_train(loss_func)
            if epoch == 1:
                print(f"Train loss: {train_loss}, Train accuracy: {train_accuracy}")
                torch.save(self.model.state_dict(), Path(self.paths.models) / (weights_name + f"epoch{epoch}.pt"))
                torch.save(self.model.state_dict(), Path(self.paths.models) / "pretrained_unet3.pt")
                self._draw_prediction(epoch)
            else:
                if train_accuracy > max(self.train_acc):
                    print(f"Train loss: {train_loss}, "
                          f"Train accuracy was improved from {max(self.train_acc)} to {train_accuracy}")
                    torch.save(self.model.state_dict(), Path(self.paths.models) / (weights_name + f"epoch{epoch}.pt"))
                    torch.save(self.model.state_dict(), Path(self.paths.models) / "pretrained_unet3.pt")
                    self._draw_prediction(epoch)
            self.train_acc.append(train_accuracy)
            self.current_epoch += 1
        print(f"The best model has been saved as: \n{str(Path(self.paths.models) / 'pretrained_unet3.pt')}")

    def predict_train(self, loss):
        acc = []
        loss_ = []
        for img, cell in zip(self.norm_data.train_image, self.norm_data.train_cells):
            train_prediction = self.model(torch.Tensor(np.expand_dims(img, axis=(0, 1))))
            train_groundtruth = torch.Tensor(np.expand_dims(cell, axis=(0, 1)))
            train_loss = loss(train_prediction, train_groundtruth)
            train_accuracy = torch.sum((train_prediction > 0.5) == train_groundtruth) / train_prediction.numel()
            acc.append(train_accuracy)
            loss_.append(train_loss.item())
        return np.mean(acc), np.mean(loss_)

    def _draw_prediction(self, step, percentile_top=99.9, percentile_bottom=0.1):
        """Draw the predictions in current step"""
        train_prediction = unet2_prediction(self.norm_data.train_image[0], self.model)
        axs = self._subplots_2images(percentile_bottom, percentile_top,
                                     self.raw_data.train_image, [train_prediction])
        axs[0].set_title("Image #1 (train)", fontdict=TITLE_STYLE)
        axs[1].set_title(f"Cell prediction #1 at step {step} (train)", fontdict=TITLE_STYLE)
        plt.tight_layout()
        plt.pause(0.1)

    def load_pretrained_unet(self, path: str):
        self.model.load_state_dict(torch.load(path))
        self.model.eval()

    def predict_test_image1(self, percentile_top=99.9, percentile_bottom=0.1):
        test_image_filenames = load_filenames(self.paths.test_image)
        test_img1_norm = (load_one_image(test_image_filenames[0]) - self.train_stat.mean) / self.train_stat.std
        test_prediction_img1 = unet2_prediction(test_img1_norm, self.model)
        axs = self._subplots_2images(percentile_bottom, percentile_top,
                                     [test_img1_norm], [test_prediction_img1])
        axs[0].set_title("Image #1 (test)", fontdict=TITLE_STYLE)
        axs[1].set_title(f"Cell prediction #1 (test)", fontdict=TITLE_STYLE)
        plt.tight_layout()
        plt.pause(0.1)

    def save_predictions_test(self):
        test_image_filenames = load_filenames(self.paths.test_image)
        with tqdm(total=len(test_image_filenames), ncols=50, unit='slice') as pbar:
            for filename in test_image_filenames:
                path_file = Path(filename)
                img_norm = (load_one_image(filename) - self.train_stat.mean) / self.train_stat.std
                prediction = unet2_prediction(img_norm, self.model)
                Image.fromarray(prediction).save(str(Path(self.paths.test_cells) / ("cell_" + path_file.stem +".tiff")))
                pbar.update(1)
