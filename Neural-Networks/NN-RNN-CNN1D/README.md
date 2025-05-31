# Time Series Prediction using Feedforward, RNN, and Conv1D Models

## Overview

This project involves predicting the values of `y` for **March 1st and March 2nd, 2020** using historical multivariate time series data (`x1`, `x2`, and `y`) ranging from **Jan 1, 2019 to Feb 29, 2020**. The data is sourced from `data.csv`.

We explore the following deep learning models:

- **Feedforward Neural Network (FNN)**
- **Recurrent Neural Network (RNN)**
- **1D Convolutional Neural Network (Conv1D)**

Each model is evaluated based on:
- Unnormalized Mean Absolute Error (MAE) on the test set
- Prediction values for March 1st and 2nd
- Loss curves (training vs validation)

In addition, we conduct a thorough **end-to-end machine learning pipeline** involving:
- Data cleaning and imputation
- Time series analysis
- Feature engineering and normalization
- Model tuning and regularization
- Evaluation and diagnostics



## Problem 1: Feedforward Neural Network 

### Results

- **Test MAE**: 6.550
- **Predicted y values**:
  - **March 1st**: y=73
  - **March 2nd**: y=71


## Problem 2: Recurrent Neural Network 

### Results

- **Test MAE**: 13.290
- **Predicted y values**:
  - **March 1st**: y=32
  - **March 2nd**: y=40


## Problem 3: 1D Convolutional Neural Network 

### Results

- **Test MAE**: 5.680
- **Predicted y values**:
  - **March 1st**: y=74
  - **March 2nd**: y=73



## Problem 4: Complete ML Workflow 

We applied the full machine learning lifecycle:

### Techniques Used

- **Data Cleaning & Imputation**:
  - Handled missing values with time-based interpolation
  - Verified continuity of date index

- **Exploratory Data Analysis**:
  - Visualized trends, seasonality, and outliers
  - Used correlation plots and lag plots

- **Feature Engineering**:
  - Lagged features, rolling means
  - Time components (day, month)

- **Normalization**:
  - MinMaxScaler on numeric features

- **Train/Test Split**:
  - Training: Jan 2019 – Jan 2020
  - Validation: Feb 2020
  - Test: March 1–2, 2020

- **Regularization & Overfitting Prevention**:
  - Early stopping
  - Dropout layers
  - Batch normalization

- **Model Selection & Hyperparameter Tuning**:
  - Grid/random search over learning rates, layer sizes, dropout rates



## Built with
1. tensorflow
2. keras
3. numpy
4. pandas
5. sklearn
6. statsmodels
7. plotly

Install dependencies:

```bash
pip install -r requirements.txt
