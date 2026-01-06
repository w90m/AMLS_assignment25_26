import medmnist
from medmnist import BreastMNIST
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import torch
from torchvision import transforms

import preprocessing

def main():
    train_dataset, val_dataset, test_dataset = preprocessing.acquire_dataset()
    preprocessing.cnn_dataset(train_dataset, val_dataset, test_dataset)



if __name__ == "__main__":
    main()
    