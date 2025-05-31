# Wine Quality Prediction and Custom Neural Network with Backpropagation

## Overview

This project contains solutions to two machine learning problems involving:
- **Problem 1**: Feature selection and prediction using Lasso Regression on `wine.csv`.
- **Problem 2**: Building and training a 3-layer neural network using backpropagation and SGD on `siCoData.csv`.

---

## Problem 1: Lasso Regression for Wine Quality 

### Objective
Predict wine quality using a reduced set of features selected via **Lasso Regression**. Evaluate model performance and compare it to a debiased version of the model.


### Approach
- Use **Lasso regression** with linear normalization.
- Use a pipeline and **GridSearchCV** (5-fold) to tune regularization parameter \( \alpha \in \text{np.logspace(-1, 0, 3)} \).
- Use **neg-median-absolute-error** as the scoring metric.
- Set seed to 42.
- Use **first 30 rows as training set**, rest as test set.

### Tasks
1. **Feature Selection via Lasso**
   - Report selected features, their coefficients, and out-of-sample error (Eout).

2. **Debias the Model**
   - Retrain a linear model on the selected features from Lasso.
   - Report new coefficients and Eout.
   - Compare debiased vs original model.

3. **Final Evaluation**
   - Plot predicted vs actual values on the test set.
   - Report final coefficients.



## Problem 2: Shallow Neural Network 

### Objective
Train a **custom 3-layer neural network** on the `siCoData.csv` dataset using:
- Backpropagation algorithm (implemented from scratch)
- Stochastic Gradient Descent (SGD)
- **tanh** activation for hidden layer, **linear** activation for output


### Training Configuration
- Loss: Mean absolute error
- Stop when:
  - Minimum in-sample error \( E_{in} \) is reached
  - OR maximum number of epochs is exceeded
- Report:
  - Minimum \( E_{in} \)
  - Final weights
  - Total number of iterations



## Built with
1. numpy
2. pandas
3. sklearn
4. matplotlib

Install dependencies:

```bash
pip install numpy pandas matplotlib scikit-learn
