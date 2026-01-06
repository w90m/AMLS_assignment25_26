import medmnist
from medmnist import BreastMNIST
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import torch
from torchvision import transforms

import gaussian_noise

def cnn_dataset_no_aug(train_dataset, val_dataset, test_dataset):
    basic_transform = transforms.Compose([
        transforms.ToTensor()
    ])

    train_dataset.transform = basic_transform
    val_dataset.transform   = basic_transform
    test_dataset.transform  = basic_transform
