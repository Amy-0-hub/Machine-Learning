import numpy as np
import pandas as pd
import random
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler
from scipy.spatial import distance_matrix
from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import random

# set the seed
random.seed(42)

#Problem 1: Parametric RBF
df = pd.read_csv('QB2022_MLR.csv')
df_test = pd.read_csv('QB2022_MLR_test.csv')
df.columns

X = df[['Pass Yds', 'Yds/Att', 'Att', 'Cmp', 'Cmp %', 'TD', 'INT']]
y = df['Rate']
scale = MinMaxScaler()
scale.fit(X)
X_norm = scale.transform(X)

# Multiple linear regression model
model1 = LinearRegression()
model1.fit(X_norm,y)
mse_train = mean_squared_error(y, model1.predict(X_norm))
print(f"Training error for linear model using LinearRegression is {mse_train}")
# Can also do
XplusBias = np.hstack([np.ones(X.shape[0]).reshape(X.shape[0],-1) , X_norm.reshape(X.shape[0],-1)])
model11 = KernelRidge(alpha=0, kernel="linear")
model11.fit(XplusBias,y)
mse_train = mean_squared_error(y, model11.predict(XplusBias))
print(f"Training error for linear model using KernelRidge is {mse_train}")

# RBF regression model
model2 = KernelRidge(alpha=0, kernel="rbf")
model2.fit(XplusBias,y)
mse_train = mean_squared_error(y, model2.predict(XplusBias))
print(f"Training error for RBF regression using KernelRidge is {mse_train}")

#predictions
## make sure to fit to train, and use that object to transform test points
X_test = df_test[['Pass Yds', 'Yds/Att', 'Att', 'Cmp', 'Cmp %', 'TD', 'INT']]
y_test = df_test['Rate']
X_norm_test = scale.transform(X_test)
XplusBias_test = np.hstack([np.ones(X_test.shape[0]).reshape(X_test.shape[0],-1) , X_norm_test.reshape(X_test.shape[0],-1)])
mse_train = mean_squared_error(y_test, model11.predict(XplusBias_test))
print(f"Test error for linear model using KernelRidge is {mse_train}")
mse_train = mean_squared_error(y_test, model2.predict(XplusBias_test))
print(f"Test error for RBF regression using KernelRidge is {mse_train}")


#Problem 2: KMeans
df = pd.read_csv("kMeansData.csv")
#normalization: data preprocessing
scalar = MinMaxScaler()
df_norm = scalar.fit_transform(df)
# visualization
df_norm = pd.DataFrame(df_norm, columns=['x1','x2'])
fig = px.scatter(df_norm, x = 'x1', y = 'x2')
fig.show()

random.seed(42)
# Initial random centeriods
# Generate random indices
random_indices = np.random.choice(df.shape[0], size= 3, replace=False)
print(random_indices)

# Select the rows corresponding to the random indices
new_centroids = df_norm.iloc[random_indices,:].values
print(new_centroids)
threshold = 0.001
delta_mu = 1
max_iter = 1_000
num_iter = 0

while(delta_mu > threshold) & (num_iter < max_iter):  
    #Distance from points to centroids
    dist = distance_matrix(df_norm[['x1','x2']], new_centroids)
    old_centroids = new_centroids
    df_norm['cluster_num'] = dist.argmin(axis=1)
    
    new_centroids = df_norm.groupby('cluster_num').mean().values
    #print(new_centroids.shape, old_centroids.shape)
    delta_mu = np.linalg.norm(new_centroids-old_centroids)
    #print(old_centroids, new_centroids, delta_mu)
    num_iter = num_iter + 1
    #print(num_iter, new_centroids)

import pandas as pd
import numpy as np
from scipy.spatial import distance_matrix

def my_kmeans_clustering(df, num_centroids = 3, threshold=1e-3, max_iter=1000):
    """
    Performs K-means clustering on a DataFrame.

    Args:
        df: Pandas DataFrame with 'x1' and 'x2' columns, normalized.
        num_centroids: The number of clusters (centroids).
        threshold: Convergence threshold for the change in centroids.
        max_iter: Maximum number of iterations.

    Returns:
        A tuple containing:
            - The DataFrame with a new 'cluster_num' column.
            - The final centroids as a NumPy array.
            - The number of iterations taken.
    """

    # Initialize centroids randomly
    num_rows = df.shape[0]
    random_indices = np.random.choice(num_rows, size=num_centroids, replace=False)
    centroids = df.iloc[random_indices,:].values

    delta_mu = 1
    num_iter = 0

    while delta_mu > threshold and num_iter < max_iter:
        # Distance from points to centroids
        dist = distance_matrix(df[['x1', 'x2']], centroids)

        # Assign cluster number based on closest centroid
        df['cluster_num'] = dist.argmin(axis=1)

        # Update centroids
        old_centroids = centroids
        centroids = df.groupby('cluster_num').mean().values

        # Calculate change in centroids
        delta_mu = np.linalg.norm(centroids - old_centroids)

        num_iter += 1
        

    return df, centroids, num_iter

df_norm = scalar.fit_transform(df)
df_norm = pd.DataFrame(df_norm, columns=['x1','x2'])
df_clustered, final_centroids, iterations = my_kmeans_clustering(df_norm, num_centroids=3)

#print(f"Final cluster assignments:\n{df_clustered}")
print(f"Final centroids:\n{final_centroids}")
print(f"Number of iterations: {iterations}")

import warnings
warnings.filterwarnings('ignore')
df_norm['cluster_num'] = df_norm['cluster_num'].astype('object')
# plot the results
fig = px.scatter(df_norm, x = 'x1', y = 'x2', color = 'cluster_num')
# Add scatter plot for centroids
fig.add_scatter(x=final_centroids[:, 0], y=final_centroids[:, 1], mode='markers', marker_size=10, showlegend = False)
fig.show()

# Compare with built-in
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=3)
kmeans.fit(df_norm[['x1','x2']])
print(kmeans.cluster_centers_)

#Problem 3: RBF Classification
rbf = pd.read_csv('rbfClassification.csv')
# Normalization
scalar = MinMaxScaler()
rbfnorm = scalar.fit_transform(rbf[['x1','x2']])
rbfnorm = pd.DataFrame(np.c_[rbfnorm, rbf.cls], columns=['x1','x2', 'cls'])

random.seed(123)
kmeans = KMeans(n_clusters=2)
kmeans.fit(rbfnorm[['x1', 'x2']])

# Report cluster centers coordinate
# centroids = kmeans.cluster_centers_
# print(centroids)
_, myCentroids, _ = my_kmeans_clustering(rbfnorm[['x1', 'x2']], num_centroids = 2, threshold=1e-3, max_iter=1000)
print(myCentroids)

from scipy.spatial import distance_matrix
X = rbfnorm[['x1', 'x2']]
y = rbfnorm['cls']
x_mu = distance_matrix(X, myCentroids)
# set the hyper-parameters
gamma = 0.5
phi = np.exp(-gamma*(x_mu)**2)
phi2 = pairwise_kernels(X, Y=myCentroids, metric='rbf',gamma = 0.5)
print(f"Printing top rows of phi to confirm they both give the same answer:\n{phi[:2]}\n{phi2[:2]}")

phiPlusOne = np.hstack([np.ones(20).reshape(20,1), phi])
w = np.dot(np.linalg.pinv(phiPlusOne),y)
print(f"RBF Network model parameters are {w}")

#predictions
# calculate the probability
pred_prob = phiPlusOne @ w.reshape(-1,1)
print(pred_prob)
# take the threshold
pred = list(map(lambda x: 1 if x>=0.5 else 0, pred_prob))
# accuracy
print(f"The model's accuracy is {sum(pred == y) / 20}")