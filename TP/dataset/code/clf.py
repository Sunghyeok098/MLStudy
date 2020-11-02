import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split

data_url = 'TP/dataset/classification/cell2celltrain.csv'
data = pd.read_csv(data_url)


print(data.head(5))

print(data.describe())
print(data.info())
print(data.isna().sum())

