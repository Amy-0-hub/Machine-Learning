# QB2022 Multiple Linear Regression Project

This project performs multiple linear regression (MLR) to predict NFL quarterback ratings for the 2022 season using statistical and machine learning techniques. It also includes a feature engineering section to examine the impact of derived features on model performance.

## Part 1: Multiple Linear Regression
- Performed regression using both:
 - Linear algebra 
 - scikit-learn
- Predicted ratings on test set and computed MSE.

## Part 2: Feature Engineering
- Created new features:
 - `Cmp/Att`: Completions per attempt
 - `TD/Att`: Touchdowns per attempt
 - `Yds/Att`: Yards per attempt
- Built model using only engineered features and compared MSE.

## Results
- Full model MSE: **39.4**
- Engineered feature model MSE: **16.3**
- **Conclusion**: The error of the second model is lower than the error of the first model that had more features.

## Deplyment
```bash
pip install numpy pandas scikit-learn