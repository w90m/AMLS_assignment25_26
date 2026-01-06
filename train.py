import medmnist
from medmnist import BreastMNIST
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import torch
from torchvision import transforms
from sklearn.svm import SVC

import gaussian_noise



def train_svm(X_train, y_train, X_val, y_val, C_value, kernel_val):
    svm = SVC(C=C_value, kernel=kernel_val, class_weight= "balanced")
    svm.fit(X_train, y_train)

    train_acc = svm.score(X_train, y_train)
    val_acc   = svm.score(X_val, y_val)

    return train_acc, val_acc



def cnn_dataset_no_aug(train_dataset, val_dataset, test_dataset):
    basic_transform = transforms.Compose([
        transforms.ToTensor()
    ])

    train_dataset.transform = basic_transform
    val_dataset.transform   = basic_transform
    test_dataset.transform  = basic_transform
