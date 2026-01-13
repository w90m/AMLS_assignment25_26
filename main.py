import medmnist
from medmnist import BreastMNIST
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import torch
from torchvision import transforms
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import precision_recall_curve

import torch.nn.functional as F

import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, classification_report
from sklearn.metrics import RocCurveDisplay, PrecisionRecallDisplay

import preprocessing, train, visualize_img

import random
import numpy as np
import torch

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False




def main():

    set_seed(42)


    #----------------------------
    # Data preparation
    #----------------------------

    train_dataset, val_dataset, test_dataset = preprocessing.acquire_dataset()
    X_train, y_train, X_val, y_val, X_test, y_test = preprocessing.train_val_test(train_dataset, val_dataset, test_dataset)

    #----------------------------
    # Visualize images 
    #----------------------------
    visualize_img.visualize(test_dataset)
    
    
    #------------------------------------------
    # SVM training - for hyperparameter tuning
    #------------------------------------------
    C_values = [0.1, 1, 10]
    kernels = ["linear", "rbf"]

    print("SVM training on raw pixel data")
    X_train_raw, X_val_raw, X_test_raw = preprocessing.svm_pipeline1(X_train, y_train, X_val, y_val, X_test, y_test)
    for k in kernels:
        print (f"Kernel: {k}")
        for C in C_values:
            train_acc, val_acc = train.train_svm(X_train_raw, y_train, X_val_raw, y_val, C_value=C, kernel_val=k)
            print(f"Train Accuracy: {train_acc:.3f}, Val Accuracy: {val_acc:.3f} for C: {C}")


    print("SVM training with PCA")
    X_train_pca, X_val_pca, X_test_pca = preprocessing.svm_pipeline2(X_train, y_train, X_val, y_val, X_test, y_test, n_components= 100)
    for k in kernels:
        print (f"Kernel: {k}")
        for C in C_values:
            train_acc, val_acc = train.train_svm(X_train_pca, y_train, X_val_pca, y_val, C_value=C, kernel_val=k)
            print(f"PCA Train Accuracy: {train_acc:.3f}, PCA Val Accuracy: {val_acc:.3f} for C: {C}")

    
    #-----------------------------------------------------------------
    # Final SVM Training and Evaluation on Test Set (selected model)
    #-----------------------------------------------------------------
    # selected the linear kernel with c =1

    # Combine train and validation sets
    X_train_full = np.concatenate([X_train_raw, X_val_raw])
    y_train_full = np.concatenate([y_train, y_val])

    final_model = train.train_final_svm(
    X_train_full,
    y_train_full,
    C_value=1,
    kernel_val="linear"
    )

    y_test_pred = final_model.predict(X_test_raw)

    acc = accuracy_score(y_test, y_test_pred)
    prec = precision_score(y_test, y_test_pred)
    rec = recall_score(y_test, y_test_pred)
    f1 = f1_score(y_test, y_test_pred)

    print("\nFinal Test Performance (SVM Without Augmentation)")
    print(f"Accuracy : {acc:.3f}")
    print(f"Precision: {prec:.3f}")
    print(f"Recall   : {rec:.3f}")
    print(f"F1-score : {f1:.3f}")

    #classification report
    # '0' is Benign, '1' is Malignant
    target_names = ['Class 0 (Benign)', 'Class 1 (Malignant)']
    print("Classification Report of SVM Without Augmentation")
    print(classification_report(y_test, y_test_pred, target_names=target_names))

    #Plots - Each in a dedicated window using 'ax'
    
    # WINDOW 1: Confusion Matrix
    fig1, ax1 = plt.subplots(figsize=(8, 6))
    ConfusionMatrixDisplay.from_predictions(
        y_test, y_test_pred, 
        display_labels=['Benign', 'Malignant'], 
        ax=ax1,
    )
    ax1.set_title("SVM Confusion Matrix (Test Set Without Augmentation)")

    # WINDOW 2: Precision-Recall Curve
    # Now uses from_predictions to automatically calculate AP (Average Precision)
    fig2, ax2 = plt.subplots(figsize=(8, 6))
    y_scores = final_model.decision_function(X_test_raw)
    
    PrecisionRecallDisplay.from_predictions(
        y_test, 
        y_scores, 
        name="SVM Without Augmentation", 
        ax=ax2
    )
    ax2.set_title('Precision-Recall Curve (SVM Test Set Without Augmentation)')
    ax2.grid(True)

    # WINDOW 3: ROC Curve
    fig3, ax3 = plt.subplots(figsize=(8, 6))
    RocCurveDisplay.from_estimator(
        final_model, 
        X_test_raw, 
        y_test, 
        ax=ax3
    )
    ax3.set_title("SVM ROC Curve (Test Set Without Augmentation)")

    # Launch all three windows at once
    plt.show()
    
    
    #-----------------------------------------------------------------
    # SVM section - WITH AUGMENTATION
    #-----------------------------------------------------------------
    print("\n" + "="*30)
    print("      SVM WITH AUGMENTATION")
    print("="*30)

    # 1. Create Augmented Training Data for SVM
    # Use the existing augmentation logic from your train.py
    # Create 2 augmented versions of every training image
    print("Augmenting SVM training set...")
    X_train_aug_list = []
    y_train_aug_list = []

    # Get the augmentation transform from train.py logic
    train_loader_aug, _, _ = train.get_dataloaders(train_dataset, val_dataset, test_dataset, augment=True)

    # Collect images from the augmented loader
    for i, (images, labels) in enumerate(train_loader_aug):
        # Convert torch tensors back to flat numpy arrays for SVM
        imgs = images.numpy().reshape(len(images), -1)
        X_train_aug_list.append(imgs)
        y_train_aug_list.append(labels.view(-1).numpy())
        if i > 5: break # Limit to a few iterations to prevent memory issues, or remove for full set

    X_train_aug = np.concatenate(X_train_aug_list)
    y_train_aug = np.concatenate(y_train_aug_list)

    # 2. Final SVM Training and Evaluation
    # selected the linear kernel with c = 10
    svm_model_aug = train.train_final_svm(
        X_train_aug,
        y_train_aug,
        C_value=10,
        kernel_val="linear"
    )

    # 3. Evaluate on Test Set
    # Using raw pixels normalized but not augmented for testing
    X_train_raw, X_val_raw, X_test_raw = preprocessing.svm_pipeline1(X_train, y_train, X_val, y_val, X_test, y_test)
    y_test_pred = svm_model_aug.predict(X_test_raw)

    acc = accuracy_score(y_test, y_test_pred)
    prec = precision_score(y_test, y_test_pred)
    rec = recall_score(y_test, y_test_pred)
    f1 = f1_score(y_test, y_test_pred)

    print("\nFinal Test Performance (Augmented SVM)")
    print(f"Accuracy : {acc:.3f}")
    print(f"Precision: {prec:.3f}")
    print(f"Recall   : {rec:.3f}")
    print(f"F1-score : {f1:.3f}")

    #classification report
    # '0' is Benign, '1' is Malignant
    target_names = ['Class 0 (Benign)', 'Class 1 (Malignant)']
    print("Classification Report of SVM Without Augmentation")
    print(classification_report(y_test, y_test_pred, target_names=target_names))

    # 4. Plots - Each in a dedicated window
    
    # WINDOW 1: Confusion Matrix
    fig1, ax1 = plt.subplots(figsize=(8, 6))
    ConfusionMatrixDisplay.from_predictions(
        y_test, y_test_pred, 
        display_labels=['Benign', 'Malignant'], 
        ax=ax1,
    )
    ax1.set_title("SVM Confusion Matrix (Augmented)")

    # WINDOW 2: Precision-Recall Curve
    fig2, ax2 = plt.subplots(figsize=(8, 6))
    y_scores = svm_model_aug.decision_function(X_test_raw)
    
    # This automatically calculates AP and puts it in the legend
    PrecisionRecallDisplay.from_predictions(
        y_test, 
        y_scores, 
        name="SVM Augmented", 
        ax=ax2
    )
    
    ax2.set_title('Precision-Recall Curve (SVM Augmented)')
    ax2.grid(True)

    # WINDOW 3: ROC Curve
    fig3, ax3 = plt.subplots(figsize=(8, 6))
    RocCurveDisplay.from_estimator(svm_model_aug, X_test_raw, y_test, ax=ax3)
    ax3.set_title("SVM ROC Curve (Augmented)")

    plt.show() # Launcher for all 3 separate windows





    #------------------------------------------------------------------
    # CNN section 
    #------------------------------------------------------------------
    
    train_loader, val_loader, test_loader = train.get_dataloaders(train_dataset, val_dataset, test_dataset, augment=False)

    # -------------------------------------------------
    # CNN Capacity × Training Budget
    # -------------------------------------------------
    
    results_to_plot = {}

    for model_type in ["Simple CNN", "Deep CNN"]:
        print(f"Training {model_type}...")
        model = train.get_cnn_model(model_type)
        # Use the modified train_cnn that returns history
        trained_model, history = train.train_cnn(model, train_loader, val_loader, epochs=200)
        results_to_plot[model_type] = history
        simple_deep_report(trained_model,test_loader )

    plot_learning_curves(results_to_plot)
    

    # -------------------------------------------------
    # CNN Adding Augmentation
    # -------------------------------------------------
    print("-------------------")
    print("Adding Augmentation")

    augmentation_results = {}
    aug_list = [False, True]
    
    for aug in aug_list:
        status = "With" if aug else "No"
        print(f"\n>>> Training Deep CNN {status} Augmentation...")
        
        # Fresh loaders for each condition
        train_loader, val_loader, test_loader = train.get_dataloaders(
            train_dataset, val_dataset, test_dataset, augment=aug
        )
        deep_model = train.get_cnn_model("Deep CNN")
        trained_model, history = train.train_cnn(deep_model, train_loader, val_loader, epochs=200)
        # Use unique keys to avoid overwriting
        augmentation_results[f"Deep CNN ({status} Aug)"] = history

        #report
        generate_final_report(deep_model, test_loader, status)

    plot_learning_curves_aug(augmentation_results)
   

def plot_learning_curves(all_results):
    """
    all_results: A dictionary where keys are model names 
    and values are the 'history' dict from train_cnn.
    """
    plt.figure(figsize=(8, 6))

    for model_name, history in all_results.items():
        epochs = range(1, len(history['val_acc']) + 1)
        plt.plot(epochs, history['val_acc'], label=f'{model_name} Val Acc')

    plt.axhline(y=0.73, color='r', linestyle='--', label='Baseline (73%)')
    plt.xlabel('Epochs')
    plt.ylabel('Validation Accuracy')
    plt.title('Learning Curves: Simple vs Deeper CNN')
    plt.legend()
    plt.grid(True)
    plt.show()



def plot_learning_curves_aug(all_results):
    """
    all_results: A dictionary where keys are aug or no aug
    and values are the 'history' dict from train_cnn.
    """
    plt.figure(figsize=(8, 6))

    for aug, history in all_results.items():
        epochs = range(1, len(history['val_acc']) + 1)
        plt.plot(epochs, history['val_acc'], label=f'{aug} Val Acc')

    plt.axhline(y=0.73, color='r', linestyle='--', label='Baseline (73%)')
    plt.xlabel('Epochs')
    plt.ylabel('Validation Accuracy')
    plt.title('Learning Curves: Deep CNN (No Augmentation) vs Deep CNN (With Augmentation)')
    plt.legend()
    plt.grid(True)
    plt.show()


def simple_deep_report(model, test_loader, device='cpu'):
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = [] 
    
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.view(-1).long()
            
            outputs = model(images)
            probs = F.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())
    
    target_names = ['Class 0 (Benign)', 'Class 1 (Malignant)']
    
    # Text Report
    print("\n" + "="*30)
    print(f"      Simple CNN vs Deep CNN REPORT ")
    print("="*30)

    # 1. Standard Metrics
    acc = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds)
    rec = recall_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)

    print(f"Accuracy : {acc:.3f}")
    print(f"Precision: {prec:.3f}")
    print(f"Recall   : {rec:.3f}")
    print(f"F1-score : {f1:.3f}")


    print(classification_report(all_labels, all_preds, target_names=target_names))
    
    
    return all_labels, all_preds, all_probs


def generate_final_report(model, test_loader, title, device='cpu'):
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = [] 
    
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.view(-1).long()
            
            outputs = model(images)
            probs = F.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())
    
    target_names = ['Class 0 (Benign)', 'Class 1 (Malignant)']
    
    # Text Report
    print("\n" + "="*30)
    print(f"      CNN FINAL REPORT: {title} Augmentation")
    print("="*30)

    # 1. Standard Metrics
    acc = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds)
    rec = recall_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)

    print(f"Accuracy : {acc:.3f}")
    print(f"Precision: {prec:.3f}")
    print(f"Recall   : {rec:.3f}")
    print(f"F1-score : {f1:.3f}")


    print(classification_report(all_labels, all_preds, target_names=target_names))
    
    # WINDOW 1: Confusion Matrix
    fig1, ax1 = plt.subplots(figsize=(8, 6))
    cm = confusion_matrix(all_labels, all_preds)
    disp_cm = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=target_names)
    disp_cm.plot(ax=ax1) 
    ax1.set_title(f'Deep CNN Confusion Matrix - {title} Augmentation')

    # WINDOW 2: ROC Curve
    fig2, ax2 = plt.subplots(figsize=(8, 6))
    RocCurveDisplay.from_predictions(
        all_labels, 
        all_probs, 
        name=f"Deep CNN - {title} Augmentation", 
        ax=ax2  # Pass ax=ax2
    )
    ax2.plot([0, 1], [0, 1], 'k--', label='Chance level (AUC = 0.5)')
    ax2.set_title(f'Deep CNN ROC Curve - {title} Augmentation')
    ax2.legend()

    # WINDOW 3: Precision-Recall Curve
    fig3, ax3 = plt.subplots(figsize=(8, 6))
    PrecisionRecallDisplay.from_predictions(
        all_labels, 
        all_probs, 
        name=f"Deep CNN - {title} Augmentation", 
        ax=ax3  # Pass ax=ax3
    )
    ax3.set_title(f'Deep CNN Precision-Recall Curve - {title} Augmentation')
    
    plt.show()
    return all_labels, all_preds, all_probs












def generate_final_report2(model, test_loader, title, device='cpu'):
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.view(-1).long() # Ensure labels are 1D
            
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())
    
    # 1. Print Text Report
    # '0' is usually 'normal/benign', '1' is 'malignant' in BreastMNIST
    target_names = ['Class 0 (Benign)', 'Class 1 (Malignant)']
    print("\n" + "="*30)
    print("      CNN FINAL REPORT")
    print("="*30)

    acc = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds)
    rec = recall_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)

    print("\nFinal Test Performance (Augmented SVM)")
    print(f"Accuracy : {acc:.3f}")
    print(f"Precision: {prec:.3f}")
    print(f"Recall   : {rec:.3f}")
    print(f"F1-score : {f1:.3f}")
    
    print(classification_report(all_labels, all_preds, target_names=target_names))
    
    # 2. Plot Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=target_names)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    disp.plot(ax=ax)
    plt.title(f'Confusion Matrix: Deep CNN {title} Augmentation')
    plt.show()

    return all_labels, all_preds
























    """
    configs = [
        ("simple", 5, False),
        ("simple", 10, False),
        ("simple", 25, False),
        ("deep",   5, False),
        ("deep",   10, False),
        ("deep",   25, False),
    ]

    results = {}
    
    for train_cnn3
    for model_type, epochs, early_stop in configs:
        print(f"\nTraining {model_type.upper()} CNN | epochs={epochs} | early_stop={early_stop}")

        model = train.get_cnn_model(model_type)

        lr = 0.001 

        trained_model = train.train_cnn(
            model,
            train_loader,
            val_loader,
            epochs=epochs,
            lr = lr,
            early_stopping=early_stop
        )

        val_acc = train.evaluate_cnn(trained_model, val_loader)
        results[(model_type, epochs, early_stop)] = val_acc

        """





def plot_phase4_results(results):
    """
    Plots validation accuracy vs epochs for Simple and Deep CNNs.
    Automatically handles missing keys in results.
    """
    models = ["simple", "deep"]

    # gather all unique epochs actually stored
    all_epochs = sorted(list(set([key[1] for key in results.keys()])))
    
    plt.figure(figsize=(8,6))

    for model in models:
        val_accs = []
        epochs_used = []
        for epochs in all_epochs:
            # try False first, else take whatever is in results
            key = (model, epochs, False)
            if key in results:
                val_acc = results[key]
            else:
                # fallback: pick any key with this model and epoch
                matching_keys = [k for k in results if k[0]==model and k[1]==epochs]
                if matching_keys:
                    val_acc = results[matching_keys[0]]
                else:
                    continue  # skip if nothing found
            val_accs.append(val_acc)
            epochs_used.append(epochs)

        plt.plot(epochs_used, val_accs, marker='o', label=f"{model.capitalize()} CNN")
    
    plt.xlabel("Training Budget (Epochs)")
    plt.ylabel("Validation Accuracy")
    plt.title("Phase 4: CNN Capacity vs Training Budget")
    plt.xticks(all_epochs)
    plt.ylim(0.5, 1.0)
    plt.grid(True)
    plt.legend()
    plt.show()





if __name__ == "__main__":
    main()
    