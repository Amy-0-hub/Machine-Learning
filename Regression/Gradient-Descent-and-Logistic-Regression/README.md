# Healthcare KNN & Logistic Regression Project

This project applies multiple machine learning and statistical techniques to healthcare and aerospace datasets. The tasks are broken into four main parts:


## Problem 1: KNN with Euclidean Distance (Using Scikit-learn)

We predict patient medication adherence (`pdc-80-flag`) using the K-Nearest Neighbors (KNN) algorithm and selected healthcare features.

### Features used:
- `pre-rx-cost`
- `numofgen`
- `numofbrand`
- `generic-cost`
- `adjust-total-30d`
- `num-er`

### Method:
- Input features normalized using linear scaling.
- Euclidean distance used.
- Tested values of k from 75 to 105 with step size 2.
- Accuracy on test set plotted and best K identified.


## Problem 2: KNN with Value Distance Metric (Custom Implementation)

We enhanced KNN by including a symbolic variable `region` using Value Distance Metric (VDM).

### Tasks:
- Calculated conditional probabilities of class labels given region.
- Computed VDM distances between `Northeast`, `Midwest`, `South`, `West`.
- Combined symbolic and numeric features in a custom KNN model.
- Compared accuracy with Problem 1.


## Problem 3: Gradient Descent Minimization

We minimize a quadratic function:

**Function:**  
> \( f(x) = x^2 + 6x \)

### Tasks:
- Found the minimum analytically using calculus.
- Implemented gradient descent to find the minimum.
- Compared both results.


## Problem 4: Logistic Regression – Challenger Disaster

We modeled the probability of O-ring failure based on launch temperature using logistic regression from scratch.

### Dataset:
- `Oring.csv` (NASA Shuttle launches prior to the Challenger disaster)

### Tasks:
- Normalized temperature using z-score.
- Implemented logistic regression using gradient descent (no built-in packages).
- Visualized model fit and calculated predicted probability of failure at 31°F.
- Discussed how engineers could have used this model in the Challenger case.

---

## Datasets

All datasets are provided in the `/data` folder:
- `healthcareTrain.csv`
- `healthcareTest.csv`
- `DataDictionary.txt`
- `Oring.csv`

---

## Built with
1. numpy
2. pandas
3. sklearn 
4. plotly
5. matplotlib
6. scipy.spatial 





