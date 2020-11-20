import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

na_values = ["None"]

data = pd.read_csv("data/cluster_data.csv", na_values=na_values, index_col='CountryCode')
income = data['IncomeGroup']
data.drop(columns=['IncomeGroup'], inplace=True)

# Data information
print(data.head())
print(data.info())
print(data.describe())
print(data.isnull().sum())

# drop NAN values
data.dropna(how='any', inplace=True)
print(data.isnull().sum())

# Scaling
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)
data_scaled = pd.DataFrame(data_scaled, columns=data.columns, index=data.index.values)
print(data_scaled)

data_scaled = pd.merge(data_scaled, income, left_index=True, right_index=True)

data_scaled.to_csv("data/pre_cluster.csv")