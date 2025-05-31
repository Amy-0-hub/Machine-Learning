import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pandas_profiling import ProfileReport
from sklearn.preprocessing import MinMaxScaler

import tensorflow as tf
from keras.models import Sequential
from keras.layers import LSTM, GRU, Dense, Conv1D, MaxPooling1D
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dropout, Flatten
from keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.optimizers import RMSprop, Adam
from keras.utils import np_utils

data = pd.read_csv('timeseriesData.csv')
profile = ProfileReport(data, title = 'Exploratory Analysis Before Cleaning', explorative =True)
profile.to_file('ExploratoryAnalysisPrecleaning.html')

#visualizing the data
fig = go.Figure()
fig.add_trace(go.Scatter(x=data['Date'],y=data['x1'],name='x1'))
fig.add_trace(go.Scatter(x=data['Date'],y=data['x2'],name='x2'))
fig.add_trace(go.Scatter(x=data['Date'],y=data['y'],name='y'))

#data processing
## data cleaning
data = data.drop(data.index[[-2,-1]])  # remove the last two rows, which all columns are NaN
# set the outliers as NaN
outlier_ind = np.argmax(data['x1'])
data.loc[outlier_ind] = data.loc[outlier_ind].replace(4230.0,np.nan).replace(99.99, np.nan).replace(517.0,np.nan)
#data.loc[outlier_ind]
date_NA_count = data.isna().sum()[0]
x1_NA_count = data.isna().sum()[1]
x2_NA_count = data.isna().sum()[2]
y_NA_count = data.isna().sum()[3]
print('Date column has %f missing values' %date_NA_count)
print('x1 column has %f missing values' %x1_NA_count)
print('x2 column has %f missing values' %x2_NA_count)
print('y column has %f missing values' %y_NA_count)

# using interpolate() method to fill missing values in time-series data
data = data.interpolate()
fig = go.Figure()
fig.add_trace(go.Scatter(x=data['Date'],y=data['x1'],name='x1'))
fig.add_trace(go.Scatter(x=data['Date'],y=data['x2'],name='x2'))
fig.add_trace(go.Scatter(x=data['Date'],y=data['y'],name='y'))

#pandas profilling report (after data cleaning)
profile2 = ProfileReport(data, title = 'Exploratory Analysis After Cleaning', explorative =True)
profile2.to_file('ExploratoryAnalysisPoscleaning.html')

# convert the y column on a given observation day into a Numpy array
y = data['y'].values
#y = data[['x1','y']].values
#print(y.shape)

##train, validation, and test splits
### train 70%, validation 15%, test 15% split
train_portion = round(y.shape[0]*0.7)
val_portion = round(y.shape[0]*0.15)

train_data = y[:train_portion]
val_data = y[train_portion:train_portion+val_portion]
test_data = y[train_portion+val_portion:]

#print('We have %d training, %d validation, and %d test data points.' % (len(train_data), len(val_data), len(test_data)))
#print(train_data.shape)
#print(val_data.shape)
#print(test_data.shape)

# normalizing the data
scaler_pred = MinMaxScaler(feature_range = (0,1))

#reshape the data to a 2D array from a scaler array
#print(train_data.shape)
train_data = train_data.reshape(-1,1)
val_data = val_data.reshape(-1,1)
test_data = test_data.reshape(-1,1)
#print(train_data.shape)
#print(val_data.shape)
#print(test_data.shape)

scaler_pred.fit(train_data)
trainNorm = scaler_pred.transform(train_data)
valNorm = scaler_pred.transform(val_data)
testNorm = scaler_pred.transform(test_data)
#print(trainNorm.shape)
#print(valNorm.shape)
#print(testNorm.shape)

# create sequences
def creatSeq(dataset, look_back, foresight):
    X, Y = [], []
    for i in range(len(dataset) - look_back - foresight):
        obs = dataset[i:(i+look_back),0]               # ,0
        X.append(obs)                                    
        Y.append(dataset[i+(look_back+foresight),0]) # ,0
    return np.array(X), np.array(Y)

trainNormX, trainNormY = creatSeq(trainNorm, look_back = 7, foresight = 1)
valNormX, valNormY = creatSeq(valNorm, look_back = 7, foresight = 1)
testNormX, testNormY = creatSeq(testNorm, look_back = 7, foresight = 1)
#print(testNormX.shape)
#print(trainNormY.shape)
#print(valNormX.shape)
#print(valNormY.shape)
#print(testNormX.shape)
#print(testNormY.shape)


# problem 1: regular feedforward neural network
#step1: define the model
model = Sequential()
#n_neurons = trainNormX.shape[1] * trainNormX.shape[2]
model.add(Dense(7, activation = 'linear', input_shape = (trainNormX.shape[1], ))) 
model.add(Dropout(0.1))
model.add(Dense(7, activation = 'linear'))
model.add(Dropout(0.1))
model.add(Dense(7, activation = 'linear'))
model.add(Dropout(0.1))
model.add(Dense(1, activation = 'linear'))
model.compile(loss = 'mae', optimizer = 'adam', metrics = ['mean_absolute_error'])

#model.summary()

#step2: fitting the model
checkpoint = EarlyStopping(monitor='val_loss',patience=5,verbose=1,mode='auto',restore_best_weights=True)
callbacks_list = [checkpoint]
network = model.fit(trainNormX,trainNormY,validation_data = (valNormX, valNormY),
          epochs=100,batch_size=64,callbacks=callbacks_list)

#step 3: error plot
# from the error plot, we need to stop the iteration at 50 to prevent overfitting
valMae = round(network.history['val_loss'][-1],2)
fig = go.Figure()
fig.add_trace(go.Scatter(y=network.history['loss'],mode='lines',name='Training Error'))
fig.add_trace(go.Scatter(y=network.history['val_loss'],mode='lines',name='Validation Error'))
fig.update_layout(yaxis_title='Mean Absolute Error',xaxis_title = 'epoch', title_text = 'Normalized MAE Validation = ' 
                  + str(valMae))
fig.show()

#step4: performance on test set
# Get the predicted values
y_pred_scaled = model.predict(testNormX)

# Unscale the predicted values
y_pred = scaler_pred.inverse_transform(y_pred_scaled)
y_test_unscaled = scaler_pred.inverse_transform(testNormY.reshape(-1, 1))

# Mean Absolute Error (MAE)
MAE = np.mean(tf.keras.metrics.mean_absolute_error(y_test_unscaled, y_pred))
print(f'Mean Absolute Error (MAE): {np.round(MAE, 2)}')

# Mean Absolute Percentage Error (MAPE)
MAPE = np.mean((np.abs(np.subtract(y_test_unscaled, y_pred)/ y_test_unscaled))) * 100
print(f'Mean Absolute Percentage Error (MAPE): {np.round(MAPE, 2)} %')

# Median Absolute Percentage Error (MDAPE)
MDAPE = np.median((np.abs(np.subtract(y_test_unscaled, y_pred)/ y_test_unscaled)) ) * 100
print(f'Median Absolute Percentage Error (MDAPE): {np.round(MDAPE, 2)} %')

fig = go.Figure()
fig.add_trace(go.Scatter(y=y_pred.reshape(-1,),mode='markers',name='Model Predictions on Test Set'))
fig.add_trace(go.Scatter(y=y_test_unscaled.reshape(-1,),mode='markers',name='Target Values for the Test Set'))
fig.update_layout(title_text = 'Unnormalized MAE Test = ' + str(MAE))
fig.show()

#step5: predicted values of y for March 1st and March 2nd
pred_y_norm = np.vstack((testNorm[-8:-1,0], testNorm[-7:,0]))
pred_y = model.predict(pred_y_norm)
pred_y_unnorm = scaler_pred.inverse_transform(pred_y)
print('y = %d on March 1st, y = %d on March 2nd' %(pred_y_unnorm[0], pred_y_unnorm[1]))


# Problem 2: Recurrent Neural Network 
## step1: preparing the data
#X = data[['x1','x2','y']].values
X = data[['x1','y']].values
#X = data[['x2','y']].values
#X.shape

trainPortion = round(X.shape[0]*0.7)
valPortion = round(X.shape[0]*0.15)

trainData = X[:trainPortion]
valData = X[trainPortion: trainPortion+valPortion]
testData = X[trainPortion+valPortion:]
print('We have %d training, %d validation, and %d test data points.' % (len(trainData), len(valData), len(testData)))

#print(trainData.shape)
#print(valData.shape)
#print(testData.shape)

sc = MinMaxScaler(feature_range = (0,1))
sc.fit(trainData)
trainNorm = sc.transform(trainData)
valNorm = sc.transform(valData)
testNorm = sc.transform(testData)

#print(trainNorm.shape)
#print(valNorm.shape)
#print(testNorm.shape)

scaler_pred = MinMaxScaler()
testNormY = testData[:,1].reshape(-1,1)
scaler_pred.fit(testNormY)
testNormYScaled = scaler_pred.transform(testNormY)
#print(testNormYScaled.shape)

def creatSeq(dataset, look_back, foresight):
    X, Y = [], []
    for i in range(len(dataset) - look_back - foresight):
        obs = dataset[i:(i+look_back)]               # sequence of 'look_back'
        X.append(obs)                                    
        Y.append(dataset[i+(look_back+foresight), 1])
    return np.array(X), np.array(Y)

trainNormX, trainNormY = creatSeq(trainNorm, 7, 2)
valNormX, valNormY = creatSeq(valNorm, 7, 2)
testNormX, testNormY = creatSeq(testNorm, 7, 2)

#print(trainNormX.shape, trainNormY.shape)
#print(valNormX.shape, valNormY.shape)
#print(testNormX.shape, testNormY.shape)

# step2: define the GRU model
model = Sequential()
n_neurons = trainNormX.shape[1] * trainNormX.shape[2]

model.add(GRU(n_neurons, input_shape=(trainNormX.shape[1], trainNormX.shape[2]), dropout=0.1, recurrent_dropout=0.1))
model.add(Dense(n_neurons, activation = 'linear'))
model.add(Dense(7, activation = 'linear'))
model.add(Dense(1, activation = 'linear'))
model.compile(loss = 'mae', optimizer = 'adam', metrics = ['mean_absolute_error'])
#model.summary()


#step3: fitting the GRU
checkpoint = EarlyStopping(monitor='val_loss',patience=5,verbose=1,mode='auto',restore_best_weights=True)
callbacks_list = [checkpoint]
network = model.fit(trainNormX,trainNormY, validation_data = (valNormX, valNormY), 
                    epochs=100,batch_size=64,callbacks=callbacks_list)



#step4: error plot
##from the error plot, we need to stop the iteration at 7 to prevent overfitting
valMae = round(network.history['val_loss'][-1],2)
fig = go.Figure()
fig.add_trace(go.Scatter(y=network.history['loss'],mode='lines',name='Training Error'))
fig.add_trace(go.Scatter(y=network.history['val_loss'],mode='lines',name='Validation Error'))
fig.update_layout(yaxis_title='Mean Absolute Error',xaxis_title = 'epoch', title_text = 'Normalized MAE Validation = ' 
                  + str(valMae))
fig.show()

#step5: performance on test set
# Get the predicted values
y_pred_scaled = model.predict(testNormX)

# Unscale the predicted values
y_pred = scaler_pred.inverse_transform(y_pred_scaled)
y_test_unscaled = scaler_pred.inverse_transform(testNormY.reshape(-1, 1))

# Mean Absolute Error (MAE)
MAE = np.mean(tf.keras.metrics.mean_absolute_error(y_test_unscaled, y_pred))
print(f'Mean Absolute Error (MAE): {np.round(MAE, 2)}')

# Mean Absolute Percentage Error (MAPE)
MAPE = np.mean((np.abs(np.subtract(y_test_unscaled, y_pred)/ y_test_unscaled))) * 100
print(f'Mean Absolute Percentage Error (MAPE): {np.round(MAPE, 2)} %')

# Median Absolute Percentage Error (MDAPE)
MDAPE = np.median((np.abs(np.subtract(y_test_unscaled, y_pred)/ y_test_unscaled)) ) * 100
print(f'Median Absolute Percentage Error (MDAPE): {np.round(MDAPE, 2)} %')

fig = go.Figure()
fig.add_trace(go.Scatter(y=y_pred.reshape(-1,),mode='markers',name='Model Predictions on Test Set'))
fig.add_trace(go.Scatter(y=y_test_unscaled.reshape(-1,),mode='markers',name='Target Values for the Test Set'))
fig.update_layout(title_text = 'Unnormalized MAE Test = ' + str(MAE))
fig.show()

#step6: predicted values of y for March 1st and March 2nd
pred_y_norm = np.vstack((testNorm[-8:-1,:], testNorm[-7:,:])).reshape(2,7,2)
pred_y_norm
pred_y = model.predict(pred_y_norm)
pred_y_unnorm = scaler_pred.inverse_transform(pred_y)
print('y = %d on March 1st, y = %d on March 2nd' %(float(pred_y_unnorm[0]), pred_y_unnorm[0]))

# Problem 3: 1D Convolutional Neural Network 
#step1: preparing the data
y = data['y'].values

train_portion = round(y.shape[0]*0.7)
val_portion = round(y.shape[0]*0.15)

train_data = y[:train_portion]
val_data = y[train_portion:train_portion+val_portion]
test_data = y[train_portion+val_portion:]

#print('We have %d training, %d validation, and %d test data points.' % (len(train_data), len(val_data), len(test_data)))
#print(train_data.shape)
#print(val_data.shape)
#print(test_data.shape)

#step2: normalizing the data
scaler_pred = MinMaxScaler(feature_range = (0,1))
#reshape the data to a 2D array from a scaler array
#print(train_data.shape)
train_data = train_data.reshape(-1,1)
val_data = val_data.reshape(-1,1)
test_data = test_data.reshape(-1,1)
#print(train_data.shape)
#print(val_data.shape)
#print(test_data.shape)

scaler_pred.fit(train_data)
trainNorm = scaler_pred.transform(train_data)
valNorm = scaler_pred.transform(val_data)
testNorm = scaler_pred.transform(test_data)
#print(trainNorm.shape)
#print(valNorm.shape)
#print(testNorm.shape)

#step3: creating sequences
def creatSeq(dataset, look_back, foresight):
    X, Y = [], []
    for i in range(len(dataset) - look_back - foresight):
        obs = dataset[i:(i+look_back),0]               # ,0
        X.append(obs)                                    
        Y.append(dataset[i+(look_back+foresight),0]) # ,0
    return np.array(X), np.array(Y)

trainNormX, trainNormY = creatSeq(trainNorm, look_back = 7, foresight = 1)
valNormX, valNormY = creatSeq(valNorm, look_back = 7, foresight = 1)
testNormX, testNormY = creatSeq(testNorm, look_back = 7, foresight = 1)
#print(testNormX.shape)
#print(trainNormY.shape)
#print(valNormX.shape)
#print(valNormY.shape)
#print(testNormX.shape)
#print(testNormY.shape)

#step4: define the model
model = Sequential()
model.add(Conv1D(64, kernel_size = 5, input_shape = (7,1), activation = 'linear'))
model.add(MaxPooling1D(pool_size = 2))
model.add(Dense(7, activation = 'linear'))
model.add(Dense(1, activation = 'linear'))
model.compile(loss = 'mae', optimizer = 'adam', metrics = ['mean_absolute_error'])
#model.summary()

#step5: fit the model
checkpoint = EarlyStopping(monitor='val_loss',patience=5,verbose=1,mode='auto',restore_best_weights=True)
callbacks_list = [checkpoint]
network = model.fit(trainNormX,trainNormY, validation_data = (valNormX, valNormY), 
                    epochs=100,batch_size=64,callbacks=callbacks_list)

#step6: error plot for Conv1D
#from the error plot, we need to stop the iteration at 58 to prevent overfitting
valMae = round(network.history['val_loss'][-1],2)
fig = go.Figure()
fig.add_trace(go.Scatter(y=network.history['loss'],mode='lines',name='Training Error'))
fig.add_trace(go.Scatter(y=network.history['val_loss'],mode='lines',name='Validation Error'))
fig.update_layout(yaxis_title='Mean Absolute Error',xaxis_title = 'epoch', title_text = 'Normalized MAE Validation = ' 
                  + str(valMae))
fig.show()

#step7: performance on test set
testNormPred = model.predict(testNormX)
testPred = scaler_pred.inverse_transform(testNormPred.reshape(-1,1))
testY = scaler_pred.inverse_transform(testNormY.reshape(-1,1))
testMae = tf.keras.metrics.mean_absolute_error(testY, testPred)

fig = go.Figure()
fig.add_trace(go.Scatter(y=testPred.reshape(-1,),mode='markers',name='Model Predictions on Test Set'))
fig.add_trace(go.Scatter(y=testY.reshape(-1,),mode='markers',name='Target Values for the Test Set'))
fig.update_layout(title_text = 'Unnormalized MAE Test = ' + str(np.mean(testMae)))
fig.show()

#step8: predicted values of y for March 1st and March 2nd
pred_y_norm = np.vstack((testNorm[-8:-1,0], testNorm[-7:,0]))
pred_y_norm
pred_y = model.predict(pred_y_norm)
pred_y_unnorm = scaler_pred.inverse_transform(pred_y.reshape(-1,1))
print('y = %d on March 1st, y = %d on March 2nd' %(float(pred_y_unnorm[0])))

# Problem 4: Complete ML Workflow 
#step1: preparing the data
X = data[['x1','x2','y']]

#step2: imports
from statsmodels.tsa.api import VAR
from statsmodels.tsa.vector_ar.var_model import VARResults
from statsmodels.tsa.stattools import grangercausalitytests

#step3: define a VAR model
model = VAR(X)

#step4: determine optimum lag order (p) by performing model fits at different lag orders to find loweset AIC
#the optimum lag order p=1
lag_order = model.select_order(15)
print(lag_order.summary())

#step5: model fit at optimum lag order and get results
results = model.fit(1)
#print(results.summary())

#the model is stationary since the absolute of all roots are greater than 1
nroots = 3*1 # k*p

print('Roots =')
for i in range(nroots):
    print(results.roots[i])

print()
print('Moduli =')
for i in range(nroots):
    print(np.absolute(results.roots[i]))

#step6: investigate granger causality between series combinations(if any)
# grangercausalitytests(X[['x1','x2']],3) # x2 did not cause x1
# grangercausalitytests(X[['x1','y']],3) # y did not cause x1
# grangercausalitytests(X[['y','x1']],3) # x1 did not cause y
# grangercausalitytests(X[['x2','x1']],3) # x1 causes x2
# grangercausalitytests(X[['y','x2']],3) # x2 causes y

#step7: predicted values of y for March 1st and March 2nd
lag_order = results.k_ar
print('Lag order =', lag_order)

forecast_values = results.forecast(X.values[-lag_order:],2)
print()
print('Forecast values:')
print(forecast_values[:,2])

