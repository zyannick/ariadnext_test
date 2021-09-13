from array import array
from builtins import type

import torch
import torch.utils.data as data_utl
import pickle as pkl

import numpy as np
import json
import csv
import h5py
import random
import os
import os.path
import glob
from random import gauss
import pandas as pd

import os
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

import torch
import torch.utils.data as data

import os
import math
import random
from os.path import *

import cv2
import time

from glob import glob

import contextlib

from scipy.signal import spectrogram

import miscellaneous.transforms_video as augmentations
from torchvision.transforms import Compose

import torchvision.transforms as transforms

DEFAULT_MEAN = (0.43216, 0.394666, 0.37645)
DEFAULT_STD = (0.22803, 0.22145, 0.216989)

# mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]


def rotate_image(image, angle):
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(image,
                            rot_mat,
                            image.shape[1::-1],
                            flags=cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_REFLECT)
    return result




def video_to_tensor(pic):
    """Convert a ``numpy.ndarray`` to tensor.
    Converts a numpy.ndarray (T x H x W x C)
    to a torch.FloatTensor of shape (C x T x H x W)

    Args:
         pic (numpy.ndarray): Video to be converted to tensor.
    Returns:
         Tensor: Converted video.
    """
    return torch.from_numpy(pic.transpose([3, 0, 1, 2]))


def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.2989, 0.5870, 0.1140])






input_size = {
    'convnet':224,
    'resnet18': 224,
    'resnet34': 224,
    'resnet50': 224,
    'vgg16': 224,
    'alexnet': 256,
    'xception': 256,
    'vit': 224
}


class Image_Reader(data_utl.Dataset):
    def __init__(self, args, filelist, transforms=None,  is_train=True):
        self.args = args
        self.filelist = filelist
        self.is_train = is_train


    def augment_data(self, img):
        r_f = np.random.randint(1, 10)
        if r_f > 5:
            img = cv2.flip(img, 1)
        return img

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        filename, label = self.filelist[index]

        img = cv2.imread(filename, 0)
        img = np.expand_dims(img, axis=2)

        img_shape = input_size[self.args.backbone]

        img = cv2.resize(img, (img_shape, img_shape))

        if self.args.augment and self.is_train:
            img = self.augment_data(img)

        label = np.asarray([label], dtype=np.float32)
        if len(img.shape) == 2:
            img = np.expand_dims(img, axis=2)
        img = np.float32(img).transpose(2, 0, 1)

        img = (img / 255.) * 2 - 1

        return torch.from_numpy(img), torch.from_numpy(label).long().squeeze()


    def __len__(self):
        return len(self.filelist)
