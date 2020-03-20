#Polynomial

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
 

#reading the dataset
dataset=pd.read_csv('coronanumber.csv')

#getting the dependent and independent columns
X=dataset.iloc[:,0:1].values
Y=dataset.iloc[:,1].values

#Splitting into the test data and the training data
"""from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,random_state=0)
"""
"""#feature scaling

from sklearn.preprocessing import StandardScaler
sc_X=StandardScaler()
X_train =sc_X.fit_transform(X_train)
X_test=sc_X.transform(X_test)
sc_Y=StandardScaler()
Y_train=sc.Y.fit_transform(Y_train)
"""

#Fitting the Linear regression
from sklearn.linear_model import LinearRegression
lin_reg=LinearRegression()
lin_reg.fit(X,Y)

#Fitting the polynomial regression

from sklearn.preprocessing import PolynomialFeatures
poly_reg= PolynomialFeatures(degree= 5)
X_poly = poly_reg.fit_transform(X)

lin_reg2=LinearRegression()
lin_reg2.fit(X_poly,Y)



#Visualising polynomial'''
'''
X_grid=np.arange(min(X),max(X),0.1)
X_grid=X_grid.reshape((len(X_grid),1))
plt.scatter(X,Y, color='red')
plt.scatter(50,lin_reg.predict([[50]]),s=150)
plt.plot(X_grid,lin_reg2.predict(poly_reg.fit_transform(X_grid)))
plt.title('Corona Cases in India')
plt.xlabel('Days')
plt.ylabel('Number of patients')
plt.show()
'''
a=lin_reg2.predict(poly_reg.fit_transform([[25]]))
print(a)

X_grid=np.arange(min(X),max(X),0.1)
X_grid=X_grid.reshape((len(X_grid),1))
plt.scatter(X,Y, color='red')
plt.scatter(25,lin_reg2.predict(poly_reg.fit_transform([[25]])),s=150)
plt.plot(X_grid,lin_reg2.predict(poly_reg.fit_transform(X_grid)))
plt.title('Corona Cases in India')
plt.xlabel('Days')
plt.ylabel('Number of patients')
plt.show()









