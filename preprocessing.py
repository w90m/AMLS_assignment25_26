import medmnist
from medmnist import BreastMNIST
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import torch
from torchvision import transforms

import gaussian_noise


def acquire_dataset():

    # Load datasets
    train_dataset = BreastMNIST(split='train', download=True)
    val_dataset   = BreastMNIST(split='val', download=True)
    test_dataset  = BreastMNIST(split='test', download=True)

    return train_dataset, val_dataset, test_dataset

def train_val_test(train_dataset, val_dataset, test_dataset):
    X_train, y_train = dataset_to_numpy(train_dataset)
    X_val, y_val     = dataset_to_numpy(val_dataset)
    X_test, y_test   = dataset_to_numpy(test_dataset)

    #print(X_train.shape)
    return X_train, y_train, X_val, y_val, X_test, y_test


def svm_pipeline1(X_train, y_train, X_val, y_val, X_test, y_test):
    #-----------------------------------------------------------------------
    # Pipeline 1 - Classical Model (SVM) with Raw Pixels
    #-----------------------------------------------------------------------
    
    #normalise pixel values. [0, 255] -> normalize to [0, 1]
    X_train_raw = X_train / 255.0
    X_val_raw   = X_val / 255.0
    X_test_raw  = X_test / 255.0

    #flatten images -- SM needs 1D feature vectors
    X_train_raw = X_train_raw.reshape(len(X_train_raw), -1)
    X_val_raw   = X_val_raw.reshape(len(X_val_raw), -1)
    X_test_raw  = X_test_raw.reshape(len(X_test_raw), -1)

    #print(X_train_raw.shape)
    return X_train_raw, X_val_raw, X_test_raw
    



def svm_pipeline2(X_train, y_train, X_val, y_val, X_test, y_test, n_components):
    #-----------------------------------------------------------------------
    # Pipeline 2 - Classical Model (SVM) with PCA Feature Extraction
    #-----------------------------------------------------------------------

    #flatten
    X_train_flat = X_train.reshape(len(X_train), -1) / 255.0
    X_val_flat   = X_val.reshape(len(X_val), -1) / 255.0
    X_test_flat  = X_test.reshape(len(X_test), -1) / 255.0

    #standardize 
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_flat)
    X_val_scaled   = scaler.transform(X_val_flat)
    X_test_scaled  = scaler.transform(X_test_flat)

    #apply PCA
    pca = PCA(n_components=100)  # 100 is a good starting point
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_val_pca   = pca.transform(X_val_scaled)
    X_test_pca  = pca.transform(X_test_scaled)

    return X_train_pca, X_val_pca, X_test_pca


#convert dataset to NumPy - for SVM

def dataset_to_numpy(dataset):
    images = []
    labels = []
    
    for img, label in dataset:
        images.append(np.array(img))
        labels.append(label)
    
    return np.array(images), np.array(labels).ravel()


















