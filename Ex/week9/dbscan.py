import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler  from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA



X = pd.read_csv('..input_path/CC_GENERAL.csv')
X = X.drop('CUST_ID', axis = 1)
X.fillna(method ='ffill', inplace = True)

scaler eStandardScaler()
X_scaled = scaler.fit_transform(X)


X_normalized = normalize(X_scaled)

pca = PCA(n_components = 2)
X_principal = pca.fit_transform(X_normalized)  
X_principal = pd.DataFrame(X_principal)  
X_principal.columns = ['P1', 'P2']  
print(X_principal.head())

db_default = DBSCAN(eps = 0.0375, min_samples =  3).fit(X_principal)
labels = db_default.labels
