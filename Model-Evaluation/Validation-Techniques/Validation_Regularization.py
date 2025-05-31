import pandas as pd
import numpy as np

# load the data 
train = pd.read_csv('healthcareTrain.csv')
train.head()

test = pd.read_csv('healthcareTest.csv')
test.head()

#problem 1: KNN with 10-Fold Cross-Validation 
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import make_column_selector as selector
from sklearn.compose import ColumnTransformer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
# To display pipeline
from sklearn import set_config 
set_config(display = 'diagram')

noncatg_pipeline = Pipeline(
    steps = [
        ('normalize', MinMaxScaler())
    ]
)
noncatg_pipeline


# Ignores the categorical features
columns_preprocessor = ColumnTransformer(
    transformers = [
        ('noncatg_transformer', noncatg_pipeline, selector(dtype_exclude = 'object')),
    ],
    remainder = 'passthrough'
)
columns_preprocessor

knn_model = Pipeline(
    steps = [
        ('preprocessor', columns_preprocessor),
        ('KNN', KNeighborsClassifier(metric = 'euclidean')),
    ]
)
knn_model

# Separating the training and test data with correct column names
x_train = train[['total_los', 'num_op', 'num_er', 'num_ndc', 'pre_total_cost', 'pre_CCI']]
y_train = train['pdc_80_flag']
x_test = test[['total_los', 'num_op', 'num_er', 'num_ndc', 'pre_total_cost', 'pre_CCI']]
y_test = test['pdc_80_flag']

from sklearn.metrics import accuracy_score

k_folds = 10
n = len(x_train)
indices = np.arange(n)

# Creating the fold
# https://www.w3schools.com/python/numpy/numpy_array_split.asp
folds = np.array_split(indices, k_folds)

# Creating a list to store the accuracy results
accuracy_results = []

# Looping through the values of k
# k = 31 to 102 with a step size of 2
for k in range(31, 102, 2):
    # Creating a list to store the accuracy results from using KFold
    kf_accuracy_results = []
    
    # Going through the 10 cross-validation folds
    for i in range(k_folds):
        # Spliting into training and validation sets for this fold
        validation_index = folds[i]

        # Creating a list to store the indices
        train_index = []
        # Going through the other folds to get the indices (not including the current one)
        for j in range(k_folds):
            if j != i:
                # Adding to the train_index list
                train_index.extend(folds[j])
        # Converting to an array
        train_index = np.array(train_index)
        
        x_train_kfold = x_train.iloc[train_index] 
        x_validation_kfold = x_train.iloc[validation_index]
        y_train_kfold = y_train.iloc[train_index]
        y_validation_kfold = y_train.iloc[validation_index]
    
        # Setting the parameters to k
        # Without it, it would only display the first value
        # https://scikit-learn.org/1.5/modules/generated/sklearn.neighbors.KNeighborsClassifier.html
        knn_model.set_params(KNN__n_neighbors = k)
    
        # Training the model on the training for this fold
        knn_model.fit(x_train_kfold, y_train_kfold)
        
        # Predicting on the validation for this fold
        y_pred_kfold = knn_model.predict(x_validation_kfold)
        # Calculating the accuracy rate between the actual test values and the predicted test values and adding it to the list
        kf_accuracy = accuracy_score(y_validation_kfold, y_pred_kfold)
        kf_accuracy_results.append(kf_accuracy)

    # Calculating the accuracy across all 10 cross-validation folds
    mean_accuracy = np.mean(kf_accuracy_results)
    accuracy_results.append({'k': k, 'Accuracy': mean_accuracy})

# Creating a df to display the MSE results
accuracy_df = pd.DataFrame(accuracy_results)
# Sorting it in descending order to get the highest accuracy
accuracy_df_best = accuracy_df.sort_values(by = 'Accuracy', ascending = False)
print(accuracy_df_best.head(1))

import plotly.express as px

# Creating a line plot for accuracy rate vs. k values
fig = px.line(
    accuracy_df,
    x = 'k',
    y = 'Accuracy',
    title = 'Accuracy Rate from 10-Fold Cross Validation vs. K Values',
    labels = {'x': 'K', 'y': 'Accuracy Rate'},
    markers = True
)

# Setting the bg color to white
fig.update_layout(
    plot_bgcolor = 'white',
    paper_bgcolor = 'white'
)

fig.show()


# Gets the best k value
best_k = accuracy_df_best.iloc[0]['k']

# Setting the parameters to the best k
knn_model.set_params(KNN__n_neighbors = int(best_k))

# Training the model using the best k
knn_model.fit(x_train, y_train)

# Predicting on the test set 
y_pred = knn_model.predict(x_test)

# Calculating the accuracy between the actual test values and the predicted test values
final_accuracy = accuracy_score(y_test, y_pred)

print(f'Accuracy with best_k:', final_accuracy)

# Gets the best accuracy value
best_accuracy = accuracy_df_best.iloc[0]['Accuracy']

# Calculating the validation and test error
validation_error = 1 - best_accuracy
test_error = 1 - final_accuracy

print(f'Validation Error: {validation_error}')
print(f'Test Error: {test_error}')



#problem 2: Bias-Variance Tradeoff 
import numpy as np
import pandas as pd
np.random.seed(42)

# BASED ON THE PREVIOUS HW SOLUTION CODE
# Creating the training points by using the ; 2 training points in R2 have a uniform distribution ranging from -1 and 1; generating 10,000 hypotheses 
df = pd.DataFrame({'x1': np.random.uniform(-1, 1, size = (10_000)),
                   'x2': np.random.uniform(-1, 1, size = (10_000))})

# Creating the target function of f(x) = sin(πx)
df['y1'] = np.sin(np.pi * df['x1'])
df['y2'] = np.sin(np.pi * df['x2'])

# Finding the slope and intercept using the Unregularized Least Squares Solution
# w = (X^T*X)^-1 * X^T*y
def unregularized(x1, x2, y1, y2):
    # Creating the design matrix X, where the first column represents the bias term
    X = np.array([[1, x1], [1, x2]])
    
    # Calculating the unregularized least squares solution
    w = np.linalg.inv(X.T @ X) @ X.T @ np.array([y1, y2])
    
    # Returning the slope and intercept
    return w

# Applying the function to get the slope and intercept for all 10,000 training sets
# Going through each row in the df
for i in range(df.shape[0]):
    # Calling the function; it calculates the slope and intercept for this particular training set
    w_unregularized = unregularized(df.loc[i, 'x1'], df.loc[i, 'x2'], df.loc[i, 'y1'], df.loc[i, 'y2'])
    # Adding the slope and intercept to the df for this particular training set
    df.loc[i, 'g_unregularized_b'], df.loc[i, 'g_unregularized_m'] = w_unregularized[0], w_unregularized[1] 

# Computing the mean values across all hypotheses for unregularized   g(x)
g_unregularized_b_bar = df['g_unregularized_b'].mean()
g_unregularized_m_bar = df['g_unregularized_m'].mean()

g_bar_unregularized = f'Unregularized Model: ḡ(x) = {g_unregularized_m_bar:.3f}x + {g_unregularized_b_bar:.3f}'
print(g_bar_unregularized)

# Finding the slope and intercept using the Regularized Least Squares Solution
# w = (X^T*X + λ*I)^-1 * X^T*y
def regularized(x1, x2, y1, y2, l = 0.1):
    # Creating the design matrix X, where the first column represents the bias term
    X = np.array([[1, x1], [1, x2]])
    
    # Setting the Identity Matrix
    # https://www.geeksforgeeks.org/numpy-identity-python/
    # Getting the number of features/columns and creating an Identity Matrix based on that
    I = np.identity(X.shape[1])
    # Not regularizing the bias because it does not contribute to the variance
    # I[0, 0] = 0

    # Calculating the regularized least squares solution
    w = np.linalg.inv(X.T @ X + l * I) @ X.T @ np.array([y1, y2])
    
    # Returning the slope and intercept
    return w

# Applying the function to get the slope and intercept for all 10,000 training sets
# Going through each row in the df
for i in range(df.shape[0]):
    # Calling the function; it calculates the slope and intercept for this particular training set
    w_regularized = regularized(df.loc[i, 'x1'], df.loc[i, 'x2'], df.loc[i, 'y1'], df.loc[i, 'y2'])
    # Adding the slope and intercept to the df for this particular training set
    df.loc[i, 'g_regularized_b'], df.loc[i, 'g_regularized_m'] = w_regularized[0], w_regularized[1] 

# Computing the mean values across all hypotheses for regularized   g(x)
g_regularized_b_bar = df['g_regularized_b'].mean()
g_regularized_m_bar = df['g_regularized_m'].mean()

g_bar_regularized = f'Regularized Model: ḡ(x) = {g_regularized_m_bar:.3f}x + {g_regularized_b_bar:.3f}'
print(g_bar_regularized)

# VERSION 1
# Evaluating g_bar at every x
df['g_unregularized_x'] = g_unregularized_b_bar + g_unregularized_m_bar * df['x1']
df['g_regularized_x'] = g_regularized_b_bar + g_regularized_m_bar * df['x1']

# bias = ḡ(x) avg pred - f(x) target values
# bias^2 = sum((ḡ(x) - f(x))^2)

# Computing the bias^2 for the unregularized model
bias2_unregularized = np.mean((df['g_unregularized_x'] - df['y1']) ** 2)

# Computing the bias^2 for the regularized model
bias2_regularized = np.mean((df['g_regularized_x'] - df['y1']) ** 2)

print(f'bias2_unregularized: {bias2_unregularized:.3f}')
print(f'bias2_regularized: {bias2_regularized:.3f}')

# VERSION 2
# Evaluating g_bar at every x using matmul
df['g_unregularized_x'] = np.matmul(np.array([g_unregularized_b_bar, g_unregularized_m_bar]), np.array([np.ones(10000), df['x1']]))
df['g_regularized_x'] = np.matmul(np.array([g_regularized_b_bar, g_regularized_m_bar]), np.array([np.ones(10000), df['x1']]))

# Computing the bias^2 for the unregularized model
bias2_unregularized_ver2 = np.mean((df['g_unregularized_x'] - df['y1']) ** 2)

# Computing the bias^2 for the regularized model
bias2_regularized_ver2 = np.mean((df['g_regularized_x'] - df['y1']) ** 2)

print(f'bias2_unregularized: {bias2_unregularized_ver2:.3f}')
print(f'bias2_regularized: {bias2_regularized_ver2:.3f}')

# VERSION 1 
# variance = E((g^D(x) - g(x))^2)
#             pred from training set - expected hypothesis (model pred) 

# gDx represents all of the hypotheses from the dataset
gDx_unregularized = pd.DataFrame(np.matmul(np.array(df[['g_unregularized_b','g_unregularized_m']]), np.array([np.ones(10000), df['x1']])))
gDx_regularized = pd.DataFrame(np.matmul(np.array(df[['g_regularized_b','g_regularized_m']]), np.array([np.ones(10000), df['x1']])))

# Computing the variance for the unregularized model
variance_unregularized = np.mean(gDx_unregularized.var())

# Computing the variance for the regularized model
variance_regularized = np.mean(gDx_regularized.var())

print(f'variance_unregularized: {variance_unregularized:.3f}')
print(f'variance_regularized: {variance_regularized:.3f}')


# VERSION 2 
# Computing the variance for the unregularized model
temp_unregularized = gDx_unregularized.sub(df['g_unregularized_x'], axis = 'columns') ** 2
varAt_x_unregularized = temp_unregularized.mean()
variance_unregularized_ver2 = np.mean(varAt_x_unregularized)

# Computing the variance for the regularized model
temp_regularized = gDx_regularized.sub(df['g_regularized_x'], axis = 'columns') ** 2
varAt_x_regularized = temp_regularized.mean()
variance_regularized_ver2 = np.mean(varAt_x_regularized)

print(f'variance_unregularized: {variance_unregularized_ver2:.3f}')
print(f'variance_regularized: {variance_regularized_ver2:.3f}')

import matplotlib.pyplot as plt

# Creating the test points
# Creating a test set of evenly spaced values of 10000 points in the range of -1 and 1
test_x = np.linspace(-1, 1, 10000)

# Getting the target values
target_values = np.sin(np.pi * test_x)

# Computing the linear model to each value in test_x for unregularized and regularized
g_bar_unregularized_values = g_unregularized_b_bar + g_unregularized_m_bar * test_x
g_bar_regularized_values = g_regularized_b_bar + g_regularized_m_bar * test_x

# Computing the gDx for the test points
gDx_unregularized = pd.DataFrame(np.matmul(np.array(df[['g_unregularized_b','g_unregularized_m']]), np.array([np.ones(10000), test_x])))
gDx_regularized = pd.DataFrame(np.matmul(np.array(df[['g_regularized_b','g_regularized_m']]), np.array([np.ones(10000), test_x])))

# Computing the variance at each test_x point for unregularized and regularized
variance_unregularized_point = np.var(gDx_unregularized, axis = 0)
variance_regularized_point = np.var(gDx_regularized, axis = 0)

# Computing the standard deviation
standard_deviation_unregularized = np.sqrt(variance_unregularized_point)
standard_deviation_regularized = np.sqrt(variance_regularized_point)

# Setting the figure size
plt.figure(figsize = (10, 6))

# Plotting the target function of f(x) = sin(πx)
plt.plot(test_x, target_values, label = 'Actual Target Function: f(x)', color = 'black')

# Plotting the average hypothesis for the unregularized model
plt.plot(test_x, g_bar_unregularized_values, label = 'ḡ(x) Unregularized', color = 'blue')

# Plotting ḡ(x) ± √var(x) for constant model which will represent by the shaded region
# ḡ(x) - √var(x) is the lower bound, ḡ(x) + √var(x) is the upper bound
plt.fill_between(test_x, g_bar_unregularized_values - standard_deviation_unregularized, g_bar_unregularized_values + standard_deviation_unregularized, 
                 label = 'ḡ(x) ± √var Unregularized', color = 'blue', alpha = 0.15)

# Plotting the average hypothesis for the regularized model
plt.plot(test_x, g_bar_regularized_values, label = 'ḡ(x) Regularized', color = 'red')

# Plotting ḡ(x) ± √var(x) for linear model which will represent by the shaded region
# ḡ(x) - √var(x) is the lower bound, ḡ(x) + √var(x) is the upper bound
plt.fill_between(test_x, g_bar_regularized_values - standard_deviation_regularized, g_bar_regularized_values + standard_deviation_regularized, 
                 label = 'ḡ(x) ± √var Regularized', color = 'red', alpha = 0.15)

# Labeling the x and y axis and adding a legend, grid, and title
plt.xlabel('x')
plt.ylabel('Function Value')
plt.title('Bias-Variance Analysis of Unregularized vs. Regularized Model')
plt.legend()
plt.grid()

# Show the plot
plt.show()

print(f'bias^2 unregularized: {bias2_unregularized:.3f}')
print(f'variance unregularized: {variance_unregularized:.3f}')
print(f'bias^2 regularized: {bias2_regularized:.3f}')
print(f'variance regularized: {variance_regularized:.3f}')

# Total Error = Variance + Bias^2

# Calculating total error for unregularized model
total_error_unregularized = variance_unregularized + bias2_unregularized

# Calculating total error for regularized model
total_error_regularized = variance_regularized + bias2_regularized

print(f'total_error_unregularized: {total_error_unregularized:.3f}')
print(f'total_error_regularized: {total_error_regularized:.3f}')