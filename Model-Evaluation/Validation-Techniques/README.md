# KNN Cross-Validation & Bias-Variance Analysis for Linear Models

## Overview

This project contains solutions for two core machine learning problems:

- **Problem 1**: Selecting the optimal `k` for K-Nearest Neighbors (KNN) using 10-fold cross-validation.
- **Problem 2**: Bias-Variance analysis for linear models with and without L2 regularization, learning the function \( f(x) = \sin(\pi x) \).



## Problem 1: KNN with 10-Fold Cross-Validation 

### Objective

- Predict the binary target variable `pdc-80-flag` using the continuous features:
  - `total-los`, `num-op`, `num-er`, `num-ndc`, `pre-total-cost`, `pre-CCI`
- Find the best value of **k** in the range [31, 101] with a step size of 2 using **custom 10-fold cross-validation**.
- Use the optimal `k` to evaluate test performance on `healthcareTest.csv`.


### Results ((Rounded to 2 Decimal Places))

| Best K  | Validation Accuracy | Test Accuracy |
|---------|---------------------|----------------|
| **31**  | **60.81%**          | **58.43%**     |



## Problem 2: Bias-Variance Tradeoff 

### Objective

Learn the function \( f(x) = \sin(\pi x) \) for \( x \in [-1, 1] \) using:
- A linear model \( h(x) = mx + b \)
- Two variants:
  - **Unregularized Linear Regression**
  - **L2-Regularized (Weight Decay, λ = 0.1)**

Each model is trained using 10,000 training sets of 2 points each (uniformly sampled).

### Tasks

1. Generate 10,000 hypotheses for both models.
2. Compute the **average hypothesis** \( \bar{g}(x) \)
3. Estimate **Bias²** and **Variance**
4. Plot \( f(x) \), \( \bar{g}(x) \), and the uncertainty band \( \bar{g}(x) \pm \sqrt{\text{Variance}(x)} \)



### Results (Rounded to 3 Decimal Places)

| Model            | Bias²   | Variance | Total Error |
|------------------|---------|----------|-------------|
| Unregularized    | 0.204   | 1.660    | 1.864       |
| L2 Regularized   | 0.230   | 0.329    | 0.559       |


## Built with
1. numpy
2. pandas
3. sklearn
4. plotly
5. matplotlib



