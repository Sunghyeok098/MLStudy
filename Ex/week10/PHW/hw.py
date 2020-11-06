import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler  
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV

data_url = 'C:/Users/hays3/git/MLStudy/Ex/dataSet/week10_dataSet\mushrooms.csv'
data = pd.read_csv(data_url)

data.replace({'?':np.nan}, inplace=True)
data = data.fillna(method='ffill')


from sklearn.preprocessing import LabelEncoder
from sklearn import preprocessing

le = LabelEncoder()
column = data.columns

for i in column:   
    data[i] = le.fit_transform(data[i])

x = data.drop(['class'], axis=1)
y = data['class']

from sklearn.cluster import DBSCAN


eps = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5]
min_samples = [5, 10, 20, 50, 100]
algorithm = ['ball_tree', 'kd_tree', 'brute']

for i in eps:
    for j in min_samples:
        for k in algorithm:

            model = DBSCAN(eps=i, min_samples=j, algorithm=k, metric='euclidean', p=2)
            predict = model.fit_predict(x)

            sum = 0

            for l in np.unique(predict):

                count0 = 0
                count1 = 0

                for o in range(len(y)):

                    if(predict[o] == l):

                        if(y[o] == 0):
                            count0 += 1
                        else:
                            count1 += 1
                
                if(count0 > count1):
                    sum = sum + count0
                else:
                    sum = sum + count1
                
                print('dbscan eps : {eps}, min_samples {min_samples}, algorithm {algorithm}' .format(eps=i, min_samples=j, algorithm=k))
                print('Purity : ', round(sum/len(y), 2))

            


x = data.drop(['class'], axis=1)
y = data['class']

#scaler = preprocessing.MinMaxScaler(feature_range=(0, 10)) 
scaler = preprocessing.MinMaxScaler() 
x = scaler.fit_transform(x) 

eps = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5]
min_samples = [5, 10, 20, 50, 100]
algorithm = ['ball_tree', 'kd_tree', 'brute']

for i in eps:
    for j in min_samples:
        for k in algorithm:

            model = DBSCAN(eps=i, min_samples=j, algorithm=k, metric='euclidean', p=2)
            predict = model.fit_predict(x)

            sum = 0

            for l in np.unique(predict):

                count0 = 0
                count1 = 0

                for o in range(len(y)):

                    if(predict[o] == l):

                        if(y[o] == 0):
                            count0 += 1
                        else:
                            count1 += 1
                
                if(count0 > count1):
                    sum = sum + count0
                else:
                    sum = sum + count1
                
                print('dbscan eps : {eps}, min_samples {min_samples}, algorithm {algorithm}' .format(eps=i, min_samples=j, algorithm=k))
                print('Purity : ', round(sum/len(y), 2))



x = data.drop(['class'], axis=1)
y = data['class']

scaler = preprocessing.StandardScaler()
x = scaler.fit_transform(x) 

eps = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5]
min_samples = [5, 10, 20, 50, 100]
algorithm = ['ball_tree', 'kd_tree', 'brute']

for i in eps:
    for j in min_samples:
        for k in algorithm:

            model = DBSCAN(eps=i, min_samples=j, algorithm=k, metric='euclidean', p=2)
            predict = model.fit_predict(x)

            sum = 0

            for l in np.unique(predict):

                count0 = 0
                count1 = 0

                for o in range(len(y)):

                    if(predict[o] == l):

                        if(y[o] == 0):
                            count0 += 1
                        else:
                            count1 += 1
                
                if(count0 > count1):
                    sum = sum + count0
                else:
                    sum = sum + count1
                
                print('dbscan eps : {eps}, min_samples {min_samples}, algorithm {algorithm}' .format(eps=i, min_samples=j, algorithm=k))
                print('Purity : ', round(sum/len(y), 2))
            

