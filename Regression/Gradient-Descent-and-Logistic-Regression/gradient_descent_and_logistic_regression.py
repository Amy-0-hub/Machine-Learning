import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import make_column_selector as selector
from sklearn.compose import ColumnTransformer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

import plotly.graph_objects as go

from sklearn import set_config #To display pipeline
set_config(display="diagram")

noncatg_pipeline = Pipeline(
    steps = [
        ("normalize", MinMaxScaler())
    ]
)
noncatg_pipeline

columns_preprocessor = ColumnTransformer(
    transformers = [
        ('noncatg_transformer', noncatg_pipeline,\
         selector(dtype_exclude = "object")),
    ],
    remainder='passthrough'
    )
columns_preprocessor

knn_model = Pipeline(
    steps = [
        ('preprocessor', columns_preprocessor),
        ( 'KNN', KNeighborsClassifier()),
    ]
)
knn_model


def knn_model(k, X_train, y_train, X_test, y_test): 
    model = Pipeline(
        steps = [
            ('preprocessor', columns_preprocessor),
            ( 'KNN', KNeighborsClassifier(n_neighbors = k)),
        ]
    )
    model.fit(X_train, y_train)
    y_test_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_test_pred)
    return(accuracy)

#Problem 1: KNN with Euclidean Distance (Using Scikit-learn)
#import data and create the model
train_data = pd.read_csv('healthcareTrain.csv')
test_data = pd.read_csv('healthcareTest.csv')
#train_data.info(verbose = True)
X_train = train_data[[ 'pre_rx_cost','numofgen','numofbrand',\
                      'generic_cost','adjust_total_30d', 'num_er']]
#The categorical data type is useful when you have a string variable consisting of only a few different values.
#Converting such a string variable to a categorical variable will save some memory,
y_train = train_data['pdc_80_flag'].astype('category')
X_test = test_data[[ 'pre_rx_cost','numofgen','numofbrand',\
                    'generic_cost','adjust_total_30d', 'num_er']]
y_test = test_data['pdc_80_flag'].astype('category')
acc_df = pd.DataFrame(columns = ['K', 'Accuracy'])
for k in range(75,106,2):
    acc = knn_model(k, X_train, y_train, X_test, y_test)
    acc_df = pd.concat([acc_df, pd.DataFrame([[k,acc]],columns = ["K","Accuracy"])])
fig = go.Figure()
fig.add_trace(go.Scatter(x = acc_df['K'], y = acc_df['Accuracy']))
fig.update_layout(xaxis_title = 'Number of Neighbors', yaxis_title = 'Accuracy')
fig.show()


# Problem 2: KNN with Value Distance Metric (Custom Implementation)
# using VDM with KNN
adTrain = pd.read_csv("healthcareTrain.csv")
adTest = pd.read_csv("healthcareTest.csv")

#Create a crosstab for region vs pdc_flag
regionTable = pd.crosstab(adTrain['regionN'], adTrain['pdc_80_flag'])
regionTable.index = ["Northeast","Midwest","South","West"]
regionTable.columns = ["pdc_flag_0", "pdc_flag_1"]
regionTable

# Find all the relevant conditional probabilities for finding VDM for symbolic variable region
# e.g P(pdc_flag_0|region = NE)
# Therefore, we need to find the sum in each row and then divide each cell by that value
regionSum = regionTable.sum(axis = 1)
print(regionSum)

prob = regionTable.divide(regionSum, axis = 0)
prob

# Use the vdm expression. e.g. delta(NE, MW) =[p(0|NE)-p(0|MW)]^2 + [p(1|NE)-p(1|MW)]^2
# This means we have to take the difference between the corresponding cells of those 2 rows.

from scipy.spatial import distance_matrix
delta = pd.DataFrame(distance_matrix(prob.values, prob.values)**2, index=prob.index, columns=prob.index)
delta

from sklearn.preprocessing import MinMaxScaler
contTitle = ["adjust_total_30d", "generic_cost","pre_rx_cost","numofgen","num_er","numofbrand"]
scaler = MinMaxScaler()
scaler.fit(adTrain[contTitle])
contTrain = pd.DataFrame(scaler.transform(adTrain[contTitle]), columns=contTitle)
contTest = pd.DataFrame(scaler.transform(adTest[contTitle]), columns=contTitle)

myDist = distance_matrix(contTrain, contTest)

totalDis = np.sqrt(myDist**2 + delta.iloc[adTrain['regionN']-1, adTest['regionN']-1])
inx= pd.DataFrame(np.argsort(totalDis, axis=0))
print(inx.shape)

predictions = []
for K in range(75, 106, 2):
    df = inx.iloc[:K, :].apply(lambda x: np.array(adTrain['pdc_80_flag'])[x])
    predictions.append(np.array(df.apply(lambda x: 1 if x.sum()> (K/2) else 0)))
predMatrix = pd.DataFrame(predictions)
predMatrix

acc = predMatrix.apply(lambda x: x == adTest['pdc_80_flag'], axis=1).mean(axis=1)
acc_df = pd.DataFrame({'K': range(75, 106, 2),'Accuracy': acc})
acc_df

fig = go.Figure()
fig.add_trace(go.Scatter(x = acc_df['K'], y = acc_df['Accuracy']))
fig.update_layout(xaxis_title = 'Number of Neighbors', yaxis_title = 'Accuracy')
fig.show()
print('K = 75 has the highest level of accuracy')


# problem 3: Gradient Descent Minimization
#initialize weight
Wo = 1

#establish number of runs
n_runs = 1000

#learning rate
alpha = 0.1

for i in range(n_runs):
    #take derivative of function for deltaF
    deriv = 2*Wo + 6
    
    #implement weight update
    new_Wo = Wo - alpha*deriv
    
    #reset wo value
    Wo = new_Wo
    
    #break statement if updated weights produce zero value derivative
    if deriv == 0:
        break
        
#print final Wo value
Wo

#Problem 4: Logistic Regression – Challenger Disaster
import matplotlib.pyplot as plt
import sklearn as skl

#loading in test and training data
oring_df = pd.read_csv('oring.csv', header =0)
oring_df.head(5)
oring_df.dtypes
oring_df.shape

# normalizing launch temperature
#add normalized temperature column
oring_df['Norm_Temp'] = (oring_df.Temp - oring_df.Temp.mean())/oring_df.Temp.std()
oring_df.head(10)

#establish sigmoid function
def sigmoid_function(x):
    sig = 1/(1 + np.exp(-x))
    return sig

#set up x and y 
x_mod = pd.DataFrame(oring_df.Norm_Temp)
y_mod = pd.DataFrame(oring_df.Failure)

#set up columns of ones for Wo
x_mod['Ones'] = pd.Series(np.ones(len(x_mod))).T

#establish number of runs
n_runs = 1000

#learning rate
alpha = 0.1

#initialize weights
w0 = 5
w1 = 15

for i in range(n_runs):
    #create matrix of weights
    w = pd.DataFrame([w1,w0])
    
    #calculate wx
    wx = np.dot(x_mod,w)

    #calculate weights using graident descent
    new_w1 = w1 - alpha*np.sum(np.dot((sigmoid_function(wx)-y_mod).T,x_mod.iloc[:,0]))
    new_w0 = w0 - alpha*np.sum(np.dot((sigmoid_function(wx)-y_mod).T,x_mod.iloc[:,1]))
    
    if(w0 - new_w0 == 0) and (w1 - new_w1 == 0):
        break
    else:
        #reset weights
        w0 = new_w0
        w1 = new_w1

#print out final weights
print('w0',w0)
print('w1',w1)

def o_ring_failure(x):
    norm_temp = (x - oring_df.Temp.mean())/oring_df.Temp.std() 
    failure_probability = 1 / (1 + np.exp(1.103 + (1.264*norm_temp)))
    return failure_probability

#create full probability series using output from o_ring_failure function
x = list(np.linspace(40,80,4000))
output = pd.Series(list(map(o_ring_failure, x)))
final = pd.concat([pd.Series(x),output],axis=1)
final.columns = ['Temperature','Probability']
final.head(5)

fig, ax = plt.subplots()
ax.scatter(oring_df.Temp, oring_df.Failure, c="red", alpha=0.5,
           label="O-Ring and Temperature")
ax.set_xlabel("Temperature")
ax.set_ylabel("Probability of Failure")
ax.set_title("O-Ring vs. Temperature")
plt.plot(final.Temperature,final.Probability)
plt.show()

o_ring_failure(31)