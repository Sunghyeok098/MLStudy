import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


def show_result(cluster, income, pca_component, title):
    colors = ['r', 'g', 'b', 'black', 'white']
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.set_title('Clustering Visualize ({})'.format(title))
    ax.set_xlabel('PC 1', fontsize=10)
    ax.set_ylabel('PC 2', fontsize=10)
    ax.set_zlabel('PC 3', fontsize=10)

    n_cluster = np.unique(cluster)
    print("{} Cluster: (income distribution)".format(title))
    for c, color in zip(n_cluster, colors):
        x = income[np.where(cluster == c)]
        _income, counts = np.unique(x, return_counts=True)
        cluster_income = dict(zip(_income, counts))
        print(cluster_income)

        ax.scatter(pca_component[np.where(cluster == c), 0],
                   pca_component[np.where(cluster == c), 1],
                   pca_component[np.where(cluster == c), 2],
                   c=color)
    plt.show()


data = pd.read_csv("data/pre_cluster.csv")
index = data['Unnamed: 0']
income = data['IncomeGroup'].to_numpy()
data.drop(columns=['Unnamed: 0', 'IncomeGroup'], inplace=True)

# PCA
pca = PCA(n_components=6)
pca_data = pca.fit_transform(data)
# print(sum(pca.explained_variance_ratio_))
xyz = PCA(n_components=3)  # 3D (x,y,z)
xyz_data = xyz.fit_transform(data)

# n_components = [1,2,3,4,5,6,7,8,9]
# variance_ratio = []
# for i in n_components:
#     p = PCA(n_components=i)
#     temp = p.fit_transform(data)
#     variance_ratio.append(sum(p.explained_variance_ratio_))
# plt.plot(n_components, variance_ratio)
# plt.xlabel('n_components')
# plt.ylabel('explained variance ratio')
# plt.show()

""" K-Means Part """
param_k = {'n_cluster': [3, 4, 5],
           'n_init': [10, 30, 50],
           'max_iter': [100, 200, 300, 400]}
kmeans = KMeans(n_clusters=param_k['n_cluster'][0], n_init=1000, max_iter=1000)
kmeans.fit(pca_data)
result = kmeans.predict(pca_data)

show_result(result, income, xyz_data, "K-Means")

""" DBSCAN PART """
param_DBSCAN = {'eps': [0.1, 0.2, 0.5, 1, 1.5, 2],
                'min_samples': [3, 5, 10, 15, 20]}
dbscan = DBSCAN(eps=param_DBSCAN['eps'][4], min_samples=param_DBSCAN['min_samples'][0])
result = dbscan.fit_predict(data)
show_result(result, income, pca_data, "DBSCAN")

# """ EM PART """
param_EM = {'n_components': [2, 3, 4, 5, 6],
            'max_iter': [50, 100, 200, 300, 1000]}
em = GaussianMixture(n_components=param_EM['n_components'][1], max_iter=param_EM['max_iter'][4])
result = em.fit_predict(pca_data)

show_result(result, income, pca_data, "EM")
