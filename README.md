# AMLS_assignment25_26
Repository for AMLS module

This project presents a comparison between a classical ML and Deep Learning approach to binary classification on BreastMNIST images.

SVM and CNN have been chosen.

Please use working method to run the file as the steps below haven't been verified due to time constrains.

Steps to running the main.py:

    Download / clone the repository..
    CD into the correct directory..
    Create an environment with: conda env create -f environment.yml..
    Activate the environment with: conda activate amls..
    Run the project with: python main.py..


The project organization:

Model_A_SVM:

    preprocessing.py:
        Contains the code to acquire and split the dataset and prepare 2 SVM pipelines

Model_B_SVM:

    gaussian_noise:
        Contains code to generate random noise

    svm_analysis:
        test code

    train:
        Contains code to train SVM, create CNNs, train and evaluate CNNs
        
    visualize:
        Contains code to view a raw image

environment.yml: 
    file that allows the creation of an enviorment with specified requirements

requirements.txt:
    file that lists all required libraries

