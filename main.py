import medmnist
from medmnist import BreastMNIST
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import torch
from torchvision import transforms

import preprocessing, train

def main():

    #----------------------------
    # Data preparation
    #----------------------------

    train_dataset, val_dataset, test_dataset = preprocessing.acquire_dataset()
    X_train, y_train, X_val, y_val, X_test, y_test = preprocessing.train_val_test(train_dataset, val_dataset, test_dataset)

    #------------------------------------------
    # SVM training - for hyperparameter tuning
    #-----------------------------------------
    C_values = [0.1, 1, 10]
    kernels = ["linear", "rbf"]

    print("SVM training on raw pixel data")
    X_train_raw, X_val_raw, X_test_raw = preprocessing.svm_pipeline1(X_train, y_train, X_val, y_val, X_test, y_test)
    for k in kernels:
        print (f"Kernel: {k}")
        for C in C_values:
            train_acc, val_acc = train.train_svm(X_train_raw, y_train, X_val_raw, y_val, C_value=C, kernel_val=k)
            print(f"Train Accuracy: {train_acc:.2f}, Val Accuracy: {val_acc:.2f} for C: {C}")


    print("SVM training with PCA")
    X_train_pca, X_val_pca, X_test_pca = preprocessing.svm_pipeline2(X_train, y_train, X_val, y_val, X_test, y_test, n_components= 100)
    for k in kernels:
        print (f"Kernel: {k}")
        for C in C_values:
            train_acc, val_acc = train.train_svm(X_train_pca, y_train, X_val_pca, y_val, C_value=C, kernel_val=k)
            print(f"PCA Train Accuracy: {train_acc:.2f}, PCA Val Accuracy: {val_acc:.2f} for C: {C}")


if __name__ == "__main__":
    main()
    