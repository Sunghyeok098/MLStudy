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


from sklearn.model_selection import GridSearchCV  
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold


print('-----------------------')
print('ridge')
ridge = Ridge()
parameters = {'alpha': [1e-15, 1e-10, 1e-8, 1e-4, 1e-3, 1e-2,  1, 5, 10, 20]}
    
ridge_regressor = GridSearchCV(ridge, parameters, scoring='neg_mean_squared_error', cv=10)  
ridge_regressor.fit(x5, y)

for i in range(len(ridge_regressor.cv_results_["params"])):
    print(ridge_regressor.cv_results_["params"][i] , "Accuracy : ", ridge_regressor.cv_results_["mean_test_score"][i])

print('Best parameter : ', ridge_regressor.best_params_)  
print('Best accuracy : ', ridge_regressor.best_score_)


print('-----------------------')
print('Lasso')
from sklearn.linear_model import Lasso 
 
lasso = Lasso()
parameters = {'alpha': [1e-15, 1e-10, 1e-8, 1e-4, 1e-3, 1e-2,  1, 5, 10, 20]}
lasso_regressor = GridSearchCV(lasso, parameters, scoring='neg_mean_squared_error', cv=10)  
lasso_regressor.fit(x5, y)

for i in range(len(lasso_regressor.cv_results_["params"])):
    print(lasso_regressor.cv_results_["params"][i] , "Accuracy : ", lasso_regressor.cv_results_["mean_test_score"][i])

print('Best parameter : ',lasso_regressor.best_params_)  
print('Best accuracy : ',lasso_regressor.best_score_)



print('-----------------------')
print('ElasticNet')
from sklearn.linear_model import ElasticNet   

elast = ElasticNet();
parameters = {'alpha': [1e-15, 1e-10, 1e-8, 1e-4, 1e-3, 1e-2,  1, 5, 10, 20], 'l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9]}
elast_regressor = GridSearchCV(elast, parameters, scoring='neg_mean_squared_error', cv=10)
elast_regressor.fit(x5, y)

for i in range(len(elast_regressor.cv_results_["params"])):
    print(elast_regressor.cv_results_["params"][i] , "Accuracy : ", elast_regressor.cv_results_["mean_test_score"][i])
 
print('Best parameter : ',elast_regressor.best_params_)  
print('Best accuracy : ',elast_regressor.best_score_)


