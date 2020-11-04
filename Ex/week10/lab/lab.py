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

data_url = 'C:/Users/hays3/git/MLStudy/Ex/dataSet/week10_dataSet\mouse.csv'
data = pd.read_csv(data_url, names=['x', 'y'])

feature = data[['x','y']]


from sklearn.cluster import DBSCAN

eps = [0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5]
min_samples = [3, 5, 10, 15, 20, 30, 50, 100]

for i in eps:
    for j in min_samples:

        model = DBSCAN(eps=i, min_samples=j)

        predict = pd.DataFrame(model.fit_predict(data))
        predict.columns=['predict']
        print(np.unique(predict))

        r = pd.concat([feature, predict],axis=1)

        fig = plt.figure( figsize=(6,6))
        plt.scatter(r['x'], r['y'], c=r['predict'],alpha=0.5)
        plt.title('dbscan eps : {eps}, min_samples {min_samples}' .format(eps=i, min_samples=j))
        plt.show()


from sklearn.cluster import KMeans

n_cluster = [2, 3, 4, 5, 6]
max_iter = [50, 100, 200, 300]

for i in n_cluster:
    for j in max_iter:

        model = KMeans(n_clusters=i, max_iter=j)

        predict = pd.DataFrame(model.fit_predict(data))
        predict.columns=['predict']
        print(np.unique(predict))

        r = pd.concat([feature, predict],axis=1)

        fig = plt.figure( figsize=(6,6))
        plt.scatter(r['x'], r['y'], c=r['predict'],alpha=0.5)
        plt.title('k-means n_clusters : {n_clusters}, max_iter {max_iter}' .format(n_clusters=i, max_iter=j))
        plt.show()
        

from sklearn import mixture

n_components = [2, 3, 4, 5, 6]
max_iter = [50, 100, 200, 300]

for i in n_components:
    for j in max_iter:

        model = mixture.GaussianMixture(n_components=i, max_iter=j)

        predict = pd.DataFrame(model.fit_predict(data))
        predict.columns=['predict']
        print(np.unique(predict))

        r = pd.concat([feature, predict],axis=1)

        fig = plt.figure( figsize=(6,6))
        plt.scatter(r['x'], r['y'], c=r['predict'],alpha=0.5)
        plt.title('EM n_components : {n_components}, max_iter {max_iter}' .format(n_components=i, max_iter=j))
        plt.show()