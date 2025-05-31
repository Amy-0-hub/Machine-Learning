# Distance-Weighted KNN Classifier

## Overview

This project implements a **distance-weighted k-nearest neighbors (KNN)** classifier from scratch using three different distance metrics:

- **L1 (Manhattan Distance)**
- **L2 (Euclidean Distance)**
- **L∞ (Chebyshev Distance)**

The goal is to classify test data points based on their similarity to the training data using the weighted votes of the nearest neighbors.


## Dataset

- **File**: `knnData.csv`
- **Features**:
  - `trainPoints_x1`, `trainPoints_x2`: Coordinates of training data points
  - `trainLabel`: Corresponding labels for training points
  - `testPoints_x1`, `testPoints_x2`: Coordinates of test data points
  - `testLabel`: Ground truth labels for test points



## Methodology

### Distance Metrics

- **L1 (Manhattan)**:  
  \[
  d = \sum |x_i - y_i|
  \]
- **L2 (Euclidean)**:  
  \[
  d = \sqrt{\sum (x_i - y_i)^2}
  \]
- **L∞ (Chebyshev)**:  
  \[
  d = \max |x_i - y_i|
  \]

### Weighted KNN

Each of the \( k \) nearest neighbors contributes a vote weighted by the inverse square of the distance:

\[
\text{weight} = \frac{1}{(d + \varepsilon)^2}
\]

where \( \varepsilon = 10^{-5} \) is added to prevent division by zero.

The predicted label is the one with the highest cumulative weight.

---

## Built with
1. numpy
2. pandas


## How to Run

1. Ensure `knnData.csv` is in the same directory.
2. Run the script:

```bash
python knn_weighted.py


