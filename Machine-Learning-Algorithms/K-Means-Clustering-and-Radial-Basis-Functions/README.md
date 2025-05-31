# Machine Learning Models on Quarterback, Clustering, and RBF Classification

This project implements various machine learning algorithms, including linear and non-linear regression models, unsupervised k-means clustering, and radial basis function (RBF) networks. The tasks are divided into three problems based on different datasets.

## Dataset
- `QB2022 MLR.csv`: Training data for quarterback rating regression.
- `QB2022 MLR test.csv`: Test data for quarterback rating regression.
- `kMeansData.csv`: Unlabeled data for unsupervised clustering.
- `rbfClassification.csv`: Labeled data for RBF-based classification.

## Problem 1: Regression with Kernel Ridge 
- Task: create two models using KernelRidge
 - Linear Kernel Regression with alpha=0
 - RBF Kernel Regression with alpha=0

- Preprocessing
 - Normalize all features using MinMaxScaler()
 - Add a column of ones to include the bias term

## Problem 2: Unsupervised K-Means Clustering 
- Task: Apply Lloyd’s k-means algorithm to cluster kMeansData.csv into 3 groups.

- Conditions
 - Random initialization of 3 cluster centers.
 - Stop when change in cluster centers < 0.001 or iteration count reaches 1000.

## Problem 3: RBF Network for Classification
- Task: Train a binary classification model using an RBF network on rbfClassification.csv.

- Steps
 - Use k-means to determine 2 RBF centers
 - Train RBF classifier using Gaussian kernel with γ = 0.5.

- Output
 - Coordinates of the two cluster centers.
 - Correct classification rate of the model.

## Built with
1. numpy
2. pandas
3. sklearn
4. plotly
5. scipy.spatial

## How to Run
pip install numpy pandas scikit-learn matplotlib


