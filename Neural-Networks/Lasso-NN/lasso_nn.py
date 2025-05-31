import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.compose import make_column_selector as selector
from sklearn.metrics import median_absolute_error
from sklearn.linear_model import LinearRegression

from sklearn.linear_model import Lasso
from sklearn import set_config

from sklearn.model_selection import train_test_split, GridSearchCV

import plotly.graph_objects as go
import matplotlib.pyplot as plt


#Problem 1: Lasso Regression for Wine Quality 
np.random.seed(42)
df = pd.read_csv('wine.csv')
df.head()

df.info(verbose = True)
df.describe()

float_pipeline = Pipeline(
    steps=[
        ("scaler", MinMaxScaler()),
    ]
)

column_transformer =  ColumnTransformer(
       transformers = [
       ('float_transformer', float_pipeline, selector(dtype_include=["float",'int64'])),
    ], verbose_feature_names_out = True #True is default
)

X = df.drop(columns = 'Quality')
y = df['Quality']

X_train = X.iloc[:30]
X_test = X.iloc[30:]
y_train = y.iloc[:30]
y_test = y.iloc[30:]
print(f"Training shape is {X_train.shape, y_train.shape}")

set_config(display="diagram")

lasso_pipe = Pipeline(
    steps = [        
        ("Preprocessing", column_transformer),
        ("LASSO", Lasso(random_state=42))        
    ]
)

lasso_params = {'LASSO__alpha':np.logspace(-1, 0, 3),
              }

import warnings
warnings.filterwarnings('ignore')
modelLasso=GridSearchCV(lasso_pipe, param_grid = lasso_params, scoring = 'neg_median_absolute_error', cv=5, refit = True)
modelLasso.fit(X_train, y_train) 

modelLasso.cv_results_

modelLasso.best_estimator_[-1].coef_

print('The coef values are:', modelLasso.best_estimator_.named_steps.LASSO.coef_)
print('The remaining features after lasso cv are:',modelLasso.feature_names_in_[modelLasso.best_estimator_.named_steps.LASSO.coef_ > 0])

#print(modelLasso.cv_results_['mean_test_score'].max())
print('Eout:', median_absolute_error(modelLasso.predict(X_test), y_test))

#debias
X_train_debias = X_train.drop(columns = ['Body', 'Oakiness', 'Clarity'])
X_test_debias = X_test.drop(columns = ['Body', 'Oakiness', 'Clarity'])

linear_pipe = Pipeline(
    steps = [        
        ("Preprocessing", column_transformer),
        ("LR", LinearRegression())        
    ]
)
modelLR = linear_pipe.fit(X_train_debias, y_train)
modelLR

print('Eout:', median_absolute_error(modelLR.predict(X_test_debias), y_test))

modelLR.named_steps['LR'].coef_

final_model = modelLasso.best_estimator_.fit(X.drop(columns = ['Clarity', 'Body', 'Oakiness']), y)
print("Final Model Coefficients:", final_model[-1].coef_)

import plotly.graph_objects as go
fig = go.Figure(go.Scatter(x = y, y = final_model.predict(X), mode = 'markers'))
fig.add_trace(go.Scatter(x = y, y = y, mode = 'lines', name = 'y=x line'))
fig.update_layout(xaxis_title = 'target', yaxis_title = 'predicted')
fig.show()

# Problem 2: Shallow Neural Network 
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
data = pd.read_csv('siCoData.csv')
x = data.x.to_numpy()
y = data.y.to_numpy()
# Preprocess the data
x = x.reshape(-1, 1)
y = y.reshape(-1, 1)

x.shape
y.shape

identity = lambda x: x # identity activation function

class deepNet(object):
  """
  Implementation of L - Layer Deep Neural Network
  """

  def __init__(self, 
                layer_dims=[], 
                initialization='random',
                regularization=None,
                alpha=None,
                stop_eps=0.001,
                optimization='gd', 
                mini_batch_size=64, 
                seed=42):
      """
      layer_dims = [X.shape[0],....]
      """
      self.params = {}
      self.grads = {}
      self.seed = seed
      self.layers = len(layer_dims)-1
      self.layer_dims = layer_dims
      self.initialization = initialization
      self.regularization = regularization
      self._lambda = alpha # regularization lambda val
      self.optimization = optimization
      self.mini_batch_size = mini_batch_size
      self.__initParams()
      self.costs = []
      self.stop_threshold = stop_eps
      # self.layers = [1, 10, 1]

  def __initParams(self):
      """
      Initialize the parameters (W[l], b[l])
      args:
      layers_dims - layers dimensions
      returns:
      params {"W[l]", "b[l]"}
      """
      np.random.seed(self.seed)
      for l in range(1, self.layers+1):
          if self.initialization == 'he':
              self.params["W" + str(l)] = np.random.randn(self.layer_dims[l],
                                                          self.layer_dims[l-1]) * np.sqrt(np.divide(2., self.layer_dims[l-1]))
              self.params["b" + str(l)] = np.zeros((self.layer_dims[l], 1))
          elif self.initialization == 'xavier':
              self.params["W" + str(l)] = np.random.randn(self.layer_dims[l],
                                                          self.layer_dims[l-1]) * np.sqrt(np.divide(1., self.layer_dims[l-1]))
              self.params["b" + str(l)] = np.zeros((self.layer_dims[l], 1))
          elif self.initialization == 'random':
              self.params["W" + str(l)] = np.random.randn(self.layer_dims[l],
                                                          self.layer_dims[l-1])
              self.params["b" + str(l)] = np.zeros((self.layer_dims[l], 1))
          print(f"W{str(l)}", self.params["W" + str(l)].shape)
          print(f"b{str(l)}", self.params["b" + str(l)].shape)
          # check dimensions
          assert (self.params["W" + str(l)].shape ==
                  (self.layer_dims[l], self.layer_dims[l-1]))
          assert (self.params["b" + str(l)].shape == (self.layer_dims[l], 1))

  def __tanh(self, Z):
      """
      Implementation of tanh(tangent hybperbolic) activation function
      args:
      Z - Linear function Z[l]
      returns:
      A - Tanh activations for current layer (l)
      """
      activation_cache = Z
      A = np.tanh(Z)
      return A, activation_cache
  
  def __identity(self, Z):
      """
      Implementation of Identity activation function
      args:
      Z - Linear function Z[l]
      returns:
      A - Identity activations for current layer (l)
      """
      activation_cache = Z
      A = Z
      return A, activation_cache
    
  def __tanh_backward(self, dA, activation_cache):
      """
      Implementation of dA * g'(Z[l]) => dA * tanh'(Z[l])
      args:
      dA - dL/dA
      activation_cache - Z (computed on forward propagation)
      returns:
      dZ - dLoss/dZ = dLoss/dA * dA/dZ => dA * dAdZ
      """
      Z = activation_cache
      dAdZ = 1 - np.power(np.tanh(Z), 2)
      dZ = np.multiply(dAdZ, dA)
      return dZ
    
  def __identity_backward(self, dA, activation_cache):
      """
      Implementation of dA * g'(Z[l]) => dA * tanh'(Z[l])
      args:
      dA - dL/dA
      activation_cache - Z (computed on forward propagation)
      returns:
      dZ - dLoss/dZ = dLoss/dA * dA/dZ => dA * dAdZ
      """
      Z = activation_cache
      dAdZ = 1
      dZ = np.multiply(dAdZ, dA)
      return dZ

  def __linear_forward(self, A_prev, W, b):
      """
      Implementation of Z = W.X + b
      args:
      A_prev - previous layer's activations (A[0] = X - input layer)
      W - weights matrix for current layer - l 
      b - bias vector for current layer - l
      returns:
      Z - linear computation for current layer - l
      linear_cache - (A_prev, W, b) for backpropagation
      """
      Z = np.dot(W, A_prev) + b
      linear_cache = (A_prev, W, b)
      # print(f"Z.shape: {Z.shape}, W.shape: {W.shape}, A_prev.shape: {A_prev.shape}, b.shape: {b.shape}")
      # check dimensions
      assert (Z.shape == (W.shape[0], A_prev.shape[1]))
      return Z, linear_cache

  def __linear_activation_forward(self, A_prev, W, b, activation):
      """
      Implementation of sigmoid(Z) for current layer
      args:
      A_prev - Activation output of previous layer (l-1)
      W - weights matrix for current layer - l
      b - bias vector for current layer - l
      returns:
      A - Activation output of current layer
      cache - (linear_cache, activation_cache)
      """
      Z, linear_cache = self.__linear_forward(A_prev, W, b)
      A, activation_cache = activation(Z)
      cache = (linear_cache, activation_cache)
      assert (A.shape == (W.shape[0], A_prev.shape[1]))
      return A, cache

  def __l_model_forward(self, X):
      """
      Implementation of forward propagation
      args:
      X or A[0] - Input features [x1 ... xn]
      returns:
      AL - Activation output of last layer (y hat)
      caches - cache values for each layer
      """
      A = X  # A[0]
      caches = []
      for l in range(1, self.layers):
          A, cache = self.__linear_activation_forward(A, 
                                                      self.params["W" + str(l)], 
                                                      self.params["b" + str(l)],
                                                      activation=self.__tanh)
          caches.append(cache)
      # for last layer - L
      AL, cache = self.__linear_activation_forward(A, 
                                                    self.params["W" + str(self.layers)], 
                                                    self.params["b" + str(self.layers)],
                                                    activation=self.__identity)
      caches.append(cache)
      # print("cache count: ", len(caches))
      return AL, caches

  def __compute_cost(self, AL, Y, Ws):
      """
      Implementation of cost function
      args:
      AL - Output of last activation (y hat)
      Y - True labels
      returns:
      cost - Cost of model at current iteration
      """
      m = Y.shape[1]
      mse = np.power((Y - AL), 2)

      if self.regularization == 'l2':
          w_sum_squares = np.sum([np.sum(np.square(w)) for w in Ws])
          l2_regularization_cost = self._lambda * w_sum_squares / (2. * m)
          cost = (1. / m) * np.sum(mse) + l2_regularization_cost
      else:
          cost = (1. / m) * np.sum(mse)

      cost = np.squeeze(cost)
      # assertions
      assert (cost.shape == ())
      return cost

  def __linear_backward(self, dZ, cache):
      """
      Implementation of dL/dW, dL/db, dL/dA_prev
      args:
      dZ - dL/dZ => dA * dA/dZ
      returns:
      db - dL/db => sum(dZ) / m
      dW - dL/dW => dZ.A_prev / m
      dA_prev - dL/dA_prev => WT.dZ / m
      """
      A_prev, W, b = cache
      m = A_prev.shape[1]
      # print(f'm is : {m}, dZ.shape is: {dZ.shape}, W.shape is: {W.shape}, A_prev.shape is: {A_prev.shape}')
      if self.regularization == 'l2':
          dA_prev = (1. / m) * np.dot(W.T, dZ)
          dW = (1. / m) * np.dot(dZ, A_prev.T) + (self._lambda / m) * W
          db = (1. / m) * np.sum(dZ, axis=1, keepdims=True) + (self._lambda / m) * b
      else:
          dA_prev = (1. / m) * np.dot(W.T, dZ)
          # m is : 1, dZ.shape is: (1, 1), W.shape is: (1, 10), A_prev.T.shape is: (1, 10)
          dW = (1. / m) * np.dot(dZ, A_prev.T)
          db = (1. / m) * np.sum(dZ, axis=1, keepdims=True)
      return dW, db, dA_prev

  def __linear_activation_backward(self, dA, cache, activation_backward):
      """
      Implementation of dL/dZ, dL/dW, dL/db, dL/dA_prev
      args:
      dA -  dL/dA for current layer
      cache - cache for current layer
      activation_backword - g'(z) function
      returns:
      db - dL/db => sum(dZ) / m
      dW - dL/dW => dZ.A_prev / m
      dA_prev - dL/dA_prev => WT.dZ / m
      """
      linear_cache, activation_cache = cache
      dZ = activation_backward(dA, activation_cache)
      dW, db, dA_prev = self.__linear_backward(dZ, linear_cache)
      return dW, db, dA_prev

  def __l_model_backward(self, AL, Y, caches):
      """
      Implementation of backpropagation for l-layer model
      args:
      AL - Last activation output
      Y - True labels
      caches - Cache of each layer params
      returns:
      grads {"dW[l]" , "db[l]", "dA_prev[l]"}
      """
      # m = AL.shape[1]
      Y = Y.reshape(AL.shape)
      # dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))
      dAL = 2 * (AL - Y)
      L = len(caches)
      curr_cache = caches[L - 1]
      dW_tmp, db_tmp, dA_prev_tmp = self.__linear_activation_backward(dAL,
                                                                      curr_cache,
                                                                      self.__identity_backward)
      self.grads["dW" + str(L)] = dW_tmp
      self.grads["db" + str(L)] = db_tmp
      self.grads["dA" + str(L-1)] = dA_prev_tmp
      # loop from self.layers-2 to layer-0
      for l in reversed(range(L-1)):
          curr_cache = caches[l]
          dW_tmp, db_tmp, dA_prev_tmp = self.__linear_activation_backward(self.grads["dA" + str(l+1)],
                                                                          curr_cache,
                                                                          self.__tanh_backward)
          self.grads["dW" + str(l+1)] = dW_tmp
          self.grads["db" + str(l+1)] = db_tmp
          self.grads["dA" + str(l)] = dA_prev_tmp

  def __update_params(self, learning_rate):
      """
      Implementation of gradient descent params update
      args:
      learning_rate - gradient descent's alpha
      returns:
      updated params
      """
      for l in range(1, self.layers):
          self.params["W" + str(l)] -= learning_rate * self.grads["dW" + str(l)]
          self.params["b" + str(l)] -= learning_rate * self.grads["db" + str(l)]

  def __optimize(self, X, Y, learning_rate):
    if len(X.shape) < 2:
      X = X.reshape(-1, 1)
    
    if len(Y.shape) < 2:
      Y = Y.reshape(-1, 1)

    AL, caches = self.__l_model_forward(X)
    Ws = [cache[0][1] for cache in caches]
    cost = self.__compute_cost(AL, Y, Ws)
    self.__l_model_backward(AL, Y, caches)
    self.__update_params(learning_rate) # update params with grads
    return cost

  def train(self, X, Y, num_iterations, learning_rate, print_cost=False):
      """
      Implementation of learning dnn model
      """
      costs = []
      m = X.shape[1] # n_features x sample_size, y.shape => 1 x sample_size
      prev_cost = 10e7
      for i in range(num_iterations):
        if self.optimization == 'gd':
            cost = self.__optimize(X, Y, learning_rate)
        elif self.optimization == 'sgd':
            for j in range(m):
                cost = self.__optimize(X[:, j], Y[:, j], learning_rate)
            #print(f"COST: {cost}")
            #costs.append(cost)
        elif self.optimization == 'mini_batch':
            permutation = [np.random.permutation(m)]
            shuffled_X = X[:, permutation]
            shuffled_Y = Y[:, permutation].reshape(1, m)
            batch_count = math.floor(m / self.mini_batch_size)
            for j in range(batch_count):
                batch_start = j * self.mini_batch_size
                batch_end = (j + 1) * self.mini_batch_size
                mini_batch_X = shuffled_X[:, batch_start:batch_end]
                mini_batch_Y = shuffled_Y[:, batch_start:batch_end]
                self.__optimize(mini_batch_X, mini_batch_Y, learning_rate)
                # mini_batch = (mini_batch_X, mini_batch_Y)
                # mini_batches.append(mini_batch)
            if m % self.mini_batch_size != 0:
                mini_batch_X = shuffled_X[:,
                                          self.mini_batch_size * batch_count:]
                mini_batch_Y = shuffled_Y[:,
                                          self.mini_batch_size * batch_count:]
                self.__optimize(mini_batch_X, mini_batch_Y, learning_rate)
        elif self.optimization == 'momentum':
            v = {}
            for l in range(1, self.layers+1):
                v["dW" + str(l)] = np.zeros((self.layer_dims[l],
                                              self.layer_dims[l-1]))
                v["db" + str(l)] = np.zeros((self.layer_dims[l],
                                              self.layer_dims[l-1]))
        # if i % 100 == 0:
        self.costs.append(cost)
        print(f"iteration: {i}, cost: {cost}")
        
        if abs(prev_cost - cost) < self.stop_threshold:
          break
        prev_cost = cost

deep_nn = deepNet(
    initialization='random',
    optimization='sgd',
    stop_eps=0.001,
    layer_dims=[x.T.shape[0], 20, 1]
)

deep_nn.train(X=x.T, Y=y.T, num_iterations=10000, learning_rate=0.01, print_cost=True)

plt.plot(list(range(0, len(deep_nn.costs))), deep_nn.costs)
plt.xlabel('iterations')
plt.ylabel('cost')
plt.show()

x.T.shape
deep_nn.params["b1"].shape
deep_nn.params["W2"].shape

A1 = np.tanh(np.dot(deep_nn.params["W1"], x.T) + deep_nn.params["b1"])
A2 = np.dot(deep_nn.params["W2"], A1) + deep_nn.params["b2"]

A2.shape
y.shape

# Plot the original data and the predicted values
plt.scatter(x, y, label="Original Data")
plt.scatter(x, A2.T, label="Predicted Values")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.show()