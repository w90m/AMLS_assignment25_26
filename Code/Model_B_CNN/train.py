import medmnist
from medmnist import BreastMNIST
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import torch
from torchvision import transforms
from sklearn.svm import SVC

import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from Code.Model_B_CNN import gaussian_noise




def train_svm(X_train, y_train, X_val, y_val, C_value, kernel_val):
    svm = SVC(C=C_value, kernel=kernel_val, class_weight= "balanced")
    svm.fit(X_train, y_train)

    train_acc = svm.score(X_train, y_train)
    val_acc   = svm.score(X_val, y_val)

    return train_acc, val_acc

def train_final_svm(X, y, C_value, kernel_val):
    model = SVC(C=C_value, kernel=kernel_val, class_weight= "balanced")
    model.fit(X, y)
    return model

#-------------------------------------------------------------------------------
# CNN Section
#-------------------------------------------------------------------------------

# SimpleCNN - the low-capacity model
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=2):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.bn1   = nn.BatchNorm2d(16)
        
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn2   = nn.BatchNorm2d(32)
        
        self.pool  = nn.MaxPool2d(2, 2)
        
        self.fc1   = nn.Linear(32*7*7, 64)
        
        #Add Dropout Layer (0.5 = 50% chance to drop)
        self.dropout = nn.Dropout(0.15) 
        
        self.fc2   = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        
        x = x.view(-1, 32*7*7)
        
        x = F.relu(self.fc1(x))
        
        #Apply Dropout before the final layer
        x = self.dropout(x) 
        
        x = self.fc2(x)
        return x

class DeeperCNN(nn.Module):
    def __init__(self, num_classes=2):
        super(DeeperCNN, self).__init__()
        # ... (Convs remain the same) ...
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.bn1   = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn2   = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn3   = nn.BatchNorm2d(64)
        self.pool  = nn.MaxPool2d(2, 2)

        self.fc1   = nn.Linear(64*3*3, 128)
        
        # CHANGE 1: Add Dropout
        self.dropout = nn.Dropout(0.5)
        
        self.fc2   = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        
        # CHANGE 2: Apply Dropout
        x = self.dropout(x)
        
        x = self.fc2(x)
        return x
    
#function used to select CNN
def get_cnn_model(version="Simple CNN"):
    if version == "Simple CNN":
        return SimpleCNN()
    elif version == "Deep CNN":
        return DeeperCNN()

#function used to tranform datasets
def cnn_dataset(train_dataset, val_dataset, test_dataset, augment=True):
    # MedMNIST images are grayscale.
    # Normalizing to mean 0.5, std 0.5 maps [0,1] to [-1,1].
    common_transforms = [
        transforms.ToTensor(),#scale the raw pixels to range 0.0 to 1.0
        transforms.Normalize(mean=[0.5], std=[0.5]) 
    ]

    if augment:
        train_transform = transforms.Compose([
            transforms.RandomRotation(degrees=10),
            transforms.RandomHorizontalFlip(), # Adding flip helps
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
            *common_transforms,
            gaussian_noise.AddGaussianNoise(0., 0.05) # Optional: Be careful with noise on small images
        ])
    else:
        train_transform = transforms.Compose(common_transforms)

    test_transform = transforms.Compose(common_transforms)

    train_dataset.transform = train_transform
    val_dataset.transform   = test_transform
    test_dataset.transform  = test_transform
    
    return None




def get_dataloaders(train_dataset, val_dataset, test_dataset, batch_size=64, augment = True):
    
    cnn_dataset(train_dataset, val_dataset, test_dataset, augment=augment)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader

#function used to train and validate the the cnn
def train_cnn(model, train_loader, val_loader, epochs=10, lr=0.001, early_stopping=False, patience=3, device='cpu'):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=5, factor=0.5)

    
    # Using the weighted loss to handle the 73% imbalance
    class_weights = torch.tensor([1.0, 3.0]).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    history = {'train_loss': [], 'val_acc': []}
    best_val_acc = 0
    best_model_state = None
    epochs_without_improvement = 0

    for epoch in range(epochs):
        # --- Training Phase ---
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.view(-1).long().to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        # --- Validation Phase ---
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.view(-1).long().to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_acc = correct / total
        avg_loss = running_loss / len(train_loader)
        
        # Store for Learning Curves
        history['train_loss'].append(avg_loss)
        history['val_acc'].append(val_acc)

        scheduler.step(val_acc)

        # ---------------------------------------------------------
        # THE PRINTOUT: Real-time information
        # ---------------------------------------------------------
        print(f"Epoch {epoch+1:02d}/{epochs} | Loss: {avg_loss:.4f} | Val Acc: {val_acc:.4f}")

        # Early stopping / Best model tracking
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict()
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if early_stopping and epochs_without_improvement >= patience:
            print(f"   --> Early stopping triggered at epoch {epoch+1}")
            break

    if best_model_state:
        model.load_state_dict(best_model_state)
    
    return model, history

#function used to evalute the cnn
def evaluate_cnn(model, test_loader, device='cpu'):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            #labels = labels.squeeze().long().to(device)
            labels = labels.view(-1).long().to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    test_acc = correct / total
    print(f"Test Accuracy: {test_acc:.4f}")
    return test_acc













#Old code


def cnn_dataset2(train_dataset, val_dataset, test_dataset, augment=True):
    """
    If augment=False, no augmentation is applied (Phase 4 experiments).
    If augment=True, augmentation is applied (Phase 5 final training).
    """

    if augment:
        train_transform = transforms.Compose([
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            gaussian_noise.AddGaussianNoise(std=0.05)
        ])
    else:
        train_transform = transforms.Compose([
            transforms.ToTensor()
        ])

    # ----------------------------------------------------
    # Validation / Test transform (NO augmentation)
    # ----------------------------------------------------
    test_transform = transforms.Compose([
        transforms.ToTensor()
    ])

    # Assign transforms
    train_dataset.transform = train_transform
    val_dataset.transform   = test_transform
    test_dataset.transform  = test_transform
    
    return None

def train_cnn2(model, train_loader, val_loader, epochs=10, lr=0.01, early_stopping = False, patience = 3, device='cpu'):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0
    epochs_without_improvement = 0

    for epoch in range(epochs):
        
        model.train()
        for images, labels in train_loader:
            images = images.to(device)
            #labels = labels.squeeze().long().to(device)
            labels = labels.view(-1).long().to(device)
            optimizer.zero_grad()
            outputs = model(images)

            """
            if epoch == 0:
                print("Labels batch:", labels[:10])
                print("Predictions:", torch.argmax(outputs, dim=1)[:10])
                break
            """
            
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        # validation
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                #labels = labels.squeeze().long().to(device)
                labels = labels.view(-1).long().to(device)

                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_acc = correct / total
        print(f"Epoch {epoch+1}/{epochs}, Validation Accuracy: {val_acc:.4f}")

        # early stopping logic (single source of truth)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model = model.state_dict()
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if early_stopping and epochs_without_improvement >= patience:
            print("Early stopping triggered")
            break


    model.load_state_dict(best_model)
    return model



def train_cnn3(model, train_loader, val_loader, epochs=10, lr=0.001, early_stopping=False, patience=3, device='cuda' if torch.cuda.is_available() else 'cpu'):
    model.to(device)
    
    # CHANGE 1: Lower LR. 0.01 is too high for Adam. Use 0.001 or 3e-4.
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4) # Added weight_decay for regularization

    # CHANGE 2: Calculate Class Weights for Imbalance
    # BreastMNIST is imbalanced. We must penalize errors on the minority class more.
    # Approximate counts: 0: ~399, 1: ~147. Weight should be inverse of frequency.
    # You can calculate this dynamically, but for BreastMNIST, these weights work well:
    class_weights = torch.tensor([1.0, 3.0]).to(device) 
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    history = {'train_loss': [], 'val_acc': []} # <--- Track history
    best_val_acc = 0
    best_model_state = None # specific variable to store state
    epochs_without_improvement = 0

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images = images.to(device)
            # MedMNIST labels are shape [batch, 1], need [batch] for CrossEntropy
            labels = labels.squeeze().long().to(device) 
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        # Validation phase
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.squeeze().long().to(device)

                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_acc = correct / total
        history['train_loss'].append(running_loss / len(train_loader))
        history['val_acc'].append(val_acc)
        
        # Print loss to ensure it's actually decreasing
        avg_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f} | Val Acc: {val_acc:.4f}")

        # Early stopping logic
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict() # Save copy
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if early_stopping and epochs_without_improvement >= patience:
            print("Early stopping triggered")
            break

    # Load best model if we saved one, otherwise current
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    return model




