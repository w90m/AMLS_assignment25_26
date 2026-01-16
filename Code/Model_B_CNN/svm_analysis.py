import matplotlib.pyplot as plt

from Code.Model_A_SVM import preprocessing
from Code.Model_B_CNN import train

def run_and_plot_svm_experiments(X_train, y_train, X_val, y_val, X_test, y_test):
    """
    Runs SVM training loops for Raw vs PCA data and plots the comparison.
    Depends on: svm_pipeline1, svm_pipeline2, train_svm
    """
    C_values = [0.1, 1, 10]
    kernels = ["linear", "rbf"]
    
    # Storage for results
    # Format: results['Type']['Kernel'] = [acc_C_0.1, acc_C_1, acc_C_10]
    results = {
        'Raw': {'linear': [], 'rbf': []},
        'PCA': {'linear': [], 'rbf': []}
    }

    print("--- Starting SVM Analysis ---")

    # -------------------------------------------------------
    # 1. Raw Data Loop
    # -------------------------------------------------------
    print("Training on Raw Data...")
    # Using your existing pipeline
    X_tr_raw, X_v_raw, X_te_raw = preprocessing.svm_pipeline1(X_train, y_train, X_val, y_val, X_test, y_test)

    for k in kernels:
        for C in C_values:
            # Using your existing train_svm function
            train_acc, val_acc = train.train_svm(X_tr_raw, y_train, X_v_raw, y_val, C_value=C, kernel_val=k)
            results['Raw'][k].append(val_acc)
            print(f"  Raw | {k} | C={C} | Val Acc: {val_acc:.4f}")

    # -------------------------------------------------------
    # 2. PCA Data Loop
    # -------------------------------------------------------
    print("\nTraining on PCA Data...")
    # Using your existing pipeline with n_components=100
    X_tr_pca, X_v_pca, X_te_pca = preprocessing.svm_pipeline2(X_train, y_train, X_val, y_val, X_test, y_test, n_components=100)

    for k in kernels:
        for C in C_values:
            train_acc, val_acc = train.train_svm(X_tr_pca, y_train, X_v_pca, y_val, C_value=C, kernel_val=k)
            results['PCA'][k].append(val_acc)
            print(f"  PCA | {k} | C={C} | Val Acc: {val_acc:.4f}")

    # -------------------------------------------------------
    # 3. Plotting
    # -------------------------------------------------------
    
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

    # Plot Raw
    for k in kernels:
        axes[0].plot(C_values, results['Raw'][k], marker='o', label=f'Kernel: {k}')
    axes[0].set_title('Raw Pixel Data')
    axes[0].set_xlabel('C Value')
    axes[0].set_ylabel('Validation Accuracy')
    axes[0].set_xscale('log') # Log scale for C values (0.1, 1, 10)
    axes[0].set_xticks(C_values)
    axes[0].set_xticklabels(C_values)
    axes[0].legend()
    axes[0].grid(True, linestyle='--', alpha=0.7)

    # Plot PCA
    for k in kernels:
        axes[1].plot(C_values, results['PCA'][k], marker='o', label=f'Kernel: {k}')
    axes[1].set_title('PCA Data (100 Components)')
    axes[1].set_xlabel('C Value')
    axes[1].set_xscale('log')
    axes[1].set_xticks(C_values)
    axes[1].set_xticklabels(C_values)
    axes[1].legend()
    axes[1].grid(True, linestyle='--', alpha=0.7)

    plt.suptitle('SVM Hyperparameter Tuning: Raw vs PCA', fontsize=16)
    plt.tight_layout()
    plt.show()