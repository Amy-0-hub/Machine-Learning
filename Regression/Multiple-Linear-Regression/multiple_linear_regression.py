import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.metrics import mean_squared_error


df = pd.read_csv("QB2022_MLR.csv")
df.head()

# part1
b=df["Rate"]
print('b shape is', b.shape, '\nColumns of ones shape is', np.ones((70,1)).shape,\
      '\nFeature data size is',df.iloc[:,2:8].shape)
A=np.hstack([np.ones((70,1)), df.iloc[:,2:9]])
print(A.shape)

x=np.linalg.inv(A.T @ A) @ A.T @ b
print('Model parameters are', x)

# make a prediction for eahc player in dataset
testdf=pd.read_csv('QB2022_MLR_test.csv')
testdf.shape
testdf.head()

testdf_withOnes = np.hstack([np.ones((testdf.shape[0],1)), testdf.iloc[:,2:9]])
print(testdf_withOnes.shape, x.shape)
testdf_pred = testdf_withOnes @ x
print(testdf_pred.shape, '\nThe predictions are', testdf_pred)

#calculate Mean Square Error for the data points
MSE_linalg = np.linalg.norm(testdf_pred - testdf['Rate'])** 2/ len(testdf['Rate'])
MSE_linalg

# using scikit-learn
import warnings
warnings.filterwarnings('ignore')
reg= linear_model.LinearRegression()
reg.fit(df.iloc[:, 2:9], df['Rate'])
print(reg.intercept_, reg.coef_)

testdf_pred = reg.predict(testdf.iloc[:,2:9])
print(testdf_pred)

MSE_sklearn = mean_squared_error(testdf_pred, testdf['Rate'])
MSE_sklearn


# part2
df['Cmp/Att'] = df['Cmp']/ df['Att']
df['TD/Att'] = df['TD']/ df['Att']

featureCols = ['Cmp/Att', 'TD/Att', 'Yds/Att']
x=df[featureCols]
reg2 = linear_model.LinearRegression()
reg2.fit(x, df['Rate'])
print('w0 is', reg2.intercept_, 'rest of ws are', reg2.coef_)

testdf['Cmp/Att'] = testdf['Cmp']/testdf['Att']
testdf['TD/Att']=testdf['TD']/testdf['Att']
testdf_pred2 = reg.predict(testdf[featureCols])
testdf_pred2

mean_squared_error(reg2.predict(testdf[featureCols]), testdf['Rate'])