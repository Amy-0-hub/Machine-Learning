# Bias-Variance Analysis

This project analyzes the bias-variance tradeoff for two hypothesis sets — **constant** and **linear** — when learning the function:

\[
f(x) = \sin(\pi x), \quad \text{for } x \in [-1, 1]
\]

Each model is trained using **two data points** sampled uniformly at random from the domain. The goal is to determine which hypothesis set better approximates the target function based on bias, variance, and the average hypothesis.

## Task Breakdown

### 1. Generate Hypotheses

- **10,000 training sets** are generated, each with two points \( (x_1, f(x_1)) \) and \( (x_2, f(x_2)) \).
- Fit models from two hypothesis sets:
  - **Constant Model**: \( h(x) = b \)
  - **Linear Model**: \( h(x) = mx + b \)

Compute the **average hypothesis** \( \bar{g}(x) \) for each model by averaging over all 10,000 hypotheses.

### 2. Compute Bias

\[
\text{Bias}^2 = \mathbb{E}_x\left[(\bar{g}(x) - f(x))^2\right]
\]

- The bias is estimated numerically using a dense grid of \( x \) values in \([-1, 1]\).

### 3. Compute Variance

\[
\text{Variance} = \mathbb{E}_x\left[\mathbb{E}_D[(g_D(x) - \bar{g}(x))^2]\right]
\]

- Compute the variance for each model using the 10,000 hypotheses and the same grid.

### 4. Plot Results

- For each hypothesis class, plot:
  - The target function \( f(x) = \sin(\pi x) \)
  - The average hypothesis \( \bar{g}(x) \)
  - The uncertainty band \( \bar{g}(x) \pm \sqrt{\text{Variance}(x)} \)

### 5. Model Selection

- Compare the models using Bias² + Variance (expected out-of-sample error).
- Select the model with the lower total error.

## Results (Rounded to 3 Decimal Places)

| Model Type   | Bias²     | Variance  | 
|--------------|-----------|-----------|
| Constant     | 0.497     | 0.247     | 
| Linear       | 0.204     | 1.66      | 



## Built witj
1. numpy
2. pandas
3. plotly


