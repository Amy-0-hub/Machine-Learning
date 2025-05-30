import pandas as pd
import numpy as np
from sklearn import linear_model

# import dataset
df= pd.read_csv("QB2022.csv")
df.head()

# linear algebra approach
b=df['Rate']
b.shape

np.array(df['Rate']).reshape(-1,1)

A= np.array([np.ones(70), df['Cmp %']])
A=A.T

x=np.linalg.inv(A.T@A) @ A.T @ b
print('Model parameters are', x[0], x[1])

# sklearn approach
reg = linear_model.LinearRegression()

reg.fit(np.array(df['Cmp %']).reshape(-1,1), np.array(df['Rate']).reshape(-1,1))
reg.intercept_, reg.coef_

# prediction
patrick = reg.predict(np.array([67.1]).reshape(-1,1))
print('His rating is', patrick)

# square error
## simple way, but not the best if at scale
print('The squared error is', (patrick-105.2)**2)



