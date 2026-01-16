import matplotlib.pyplot as plt
from medmnist import BreastMNIST
import numpy as np


def visualize(dataset):

    # Load test split (no transforms, raw images)
    #dataset = BreastMNIST(split='test', download=True)

    benign_img = None
    malignant_img = None

    # Find one sample of each class
    for img, label in dataset:
        label = int(label)
        if label == 0 and benign_img is None:
            benign_img = img
        elif label == 1 and malignant_img is None:
            malignant_img = img
        
        if benign_img is not None and malignant_img is not None:
            break

    # Plot the images
    plt.figure(figsize=(6, 3))

    plt.subplot(1, 2, 1)
    plt.imshow(benign_img, cmap='gray')
    plt.title("Benign (Label 0)")
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(malignant_img, cmap='gray')
    plt.title("Malignant (Label 1)")
    plt.axis('off')

    plt.tight_layout()
    plt.show()
