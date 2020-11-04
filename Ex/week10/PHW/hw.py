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

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

column = data.columns



for i in column:
    data[i] = le.fit_transform(data[i])

print(data)

"""
data['buying'] = le.fit_transform(data['buying'])
data['maint'] = le.fit_transform(data['maint'])
data['doors'] = le.fit_transform(data['doors'])
data['persons'] = le.fit_transform(data['persons'])
data['lug_boot'] = le.fit_transform(data['lug_boot'])
data['safety'] = le.fit_transform(data['safety'])
data['car'] = le.fit_transform(data['car'])


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

"""