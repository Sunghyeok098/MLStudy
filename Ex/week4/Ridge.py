import pandas as pd  
import numpy as np
import matplotlib.pyplot as plt

data_url = 'Ex\week4\insurance.csv'
data = pd.read_csv(data_url)

def scatter_plot(feature, target):
    plt.figure(figsize=((16, 8)))
    plt.scatter(data[feature],  data[target],  c='black')
    #plt.show()


scatter_plot('age', 'charges')  
scatter_plot('sex','charges')  
scatter_plot('bmi', 'charges')
scatter_plot('children', 'charges')
scatter_plot('smoker', 'charges')
scatter_plot('region', 'charges')

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
print(mean_MSE)


from sklearn.model_selection import GridSearchCV  
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold

kfold = KFold(10)
for train, test in kfold.split(data):
    
    data_train = data.loc[[i for i in train], :]
    
    x5 = data_train.drop(['charges'], axis=1)
    y = data_train['charges']
    
    ridge = Ridge()
    parameters = {'alpha': [1e-15, 1e-10, 1e-8, 1e-4, 1e-3, 1e-2,  1, 5, 10, 20]}
    
    ridge_regressor = GridSearchCV(ridge, parameters, scoring='neg_mean_squared_error', cv=10)  
    ridge_regressor.fit(x5, y)

    print(ridge_regressor.best_params_)  
    print(ridge_regressor.best_score_)

    

 
    


