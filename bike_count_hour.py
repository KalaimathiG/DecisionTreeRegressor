# -*- coding: utf-8 -*-
"""
Created on Fri Oct  9 17:22:44 2020

@author: 591664
"""


# -*- coding: utf-8 -*-
"""
Created on Thu Oct  8 19:36:12 2020

@author: 591664
"""


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot  as plt

bike_data=pd.read_csv('C:\\Users\\591664\\Downloads\\Github projects\\Decision tree regression\\archive\\hour.csv')

plt.title('Season Vs Count')
sns.barplot(bike_data['season'],bike_data['cnt'])
plt.show()

fig_dims = (15,4)
fig, ax = plt.subplots(figsize=fig_dims)
bike_data.groupby(['yr','mnth'])['cnt'].sum().plot(kind='bar',ax=ax)
plt.xlabel(' Month')
plt.ylabel('Count')
plt.title('Monthwise count for 2011,2012')
plt.show()



plt.title('Days Vs Counts')
sns.pairplot(bike_data,x_vars=['holiday','weekday','workingday'],y_vars='cnt')

plt.title('WindSpeed vs Count')
sns.scatterplot(x=bike_data['windspeed'],y=bike_data['cnt'])


sns.pairplot(bike_data,x_vars=['temp','atemp','hum'],y_vars='cnt')

plt.title('weathersit Vs Count')
sns.barplot(bike_data['weathersit'],bike_data['cnt'])
plt.show()

sns.heatmap(bike_data.corr(),annot=True,cmap='winter',linewidths=0.25,linecolor='magenta')

X=bike_data.iloc[:,2:13].values
y=bike_data.iloc[:,-1].values

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y)

from sklearn.tree import DecisionTreeRegressor
dtr=DecisionTreeRegressor()
dtr.fit(X_train,y_train)

y_pred=dtr.predict(X_test)

from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error
print("R square :",r2_score(y_test,y_pred))
print("MAE :",mean_absolute_error(y_test,y_pred))
print("MSE :",mean_squared_error(y_test,y_pred))
print("RMSE :",mean_squared_error(y_test,y_pred)**0.5)

from sklearn.linear_model import LinearRegression
lr=LinearRegression()
lr.fit(X,y)
y_pred2=lr.predict(X_test)

from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error
print("R square :",r2_score(y_test,y_pred2))
print("MAE :",mean_absolute_error(y_test,y_pred2))
print("MSE :",mean_squared_error(y_test,y_pred2))
print("RMSE :",mean_squared_error(y_test,y_pred2)**0.5)
