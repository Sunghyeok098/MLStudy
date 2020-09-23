import pandas as pd  
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings(action='ignore') 

data_url = 'Ex\week4\insurance.csv'
data = pd.read_csv(data_url)


from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
data['sex'] = le.fit_transform(data['sex'])
data['smoker'] = le.fit_transform(data['smoker'])
data['region'] = le.fit_transform(data['region'])


from sklearn.model_selection import cross_val_score  
from sklearn.linear_model import LinearRegression  

x5 = data.drop(['charges'], axis=1)
y = data['charges']


lin_reg = LinearRegression()
MSE5 = cross_val_score(lin_reg, x5, y,  scoring='neg_mean_squared_error', cv=5)
mean_MSE = np.mean(MSE5)  
print('linear regreesion : ')
print(mean_MSE)
print('\n')


from sklearn.model_selection import GridSearchCV  
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold


print('ridge')
ridge = Ridge()
parameters = {'alpha': [1e-15, 1e-10, 1e-8, 1e-4, 1e-3, 1e-2,  1, 5, 10, 20]}
    
ridge_regressor = GridSearchCV(ridge, parameters, scoring='neg_mean_squared_error', cv=10)  
ridge_regressor.fit(x5, y)

print(ridge_regressor.best_params_)  
print(ridge_regressor.best_score_)
print('\n')


print('Lasso')
from sklearn.linear_model import Lasso 
 
lasso = Lasso()
parameters = {'alpha': [1e-15, 1e-10, 1e-8, 1e-4, 1e-3, 1e-2,  1, 5, 10, 20]}
lasso_regressor = GridSearchCV(lasso, parameters, scoring='neg_mean_squared_error', cv=10)  
lasso_regressor.fit(x5, y)


print(lasso_regressor.best_params_)  
print(lasso_regressor.best_score_)
print('\n')



print('ElasticNet')
from sklearn.linear_model import ElasticNet   

elast = ElasticNet();
parameters = {'alpha': [1e-15, 1e-10, 1e-8, 1e-4, 1e-3, 1e-2,  1, 5, 10, 20], 'l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9]}
elast_regressor = GridSearchCV(elast, parameters, scoring='neg_mean_squared_error', cv=10)
elast_regressor.fit(x5, y)
 
print(elast_regressor.best_params_)  
print(elast_regressor.best_score_)


