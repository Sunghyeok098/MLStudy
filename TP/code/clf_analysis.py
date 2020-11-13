import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.metrics import classification_report,  confusion_matrix
from sklearn.model_selection import train_test_split, GridSearchCV
from mpl_toolkits.mplot3d import axes3d
from matplotlib import style

import warnings
warnings.filterwarnings(action='ignore')
pd.set_option('display.max_row', 500)
pd.set_option('display.max_columns', 100)

data_url = 'TP/dataset/classification/labeling_data.csv'
data = pd.read_csv(data_url)


print("\nStart RandomForest")

data_train, data_test = train_test_split(data, test_size=0.2)

x_train = data_train.drop(['Churn'], axis=1)
y_train = data_train['Churn']
x_test = data_test.drop(['Churn'], axis=1)
y_test = data_test['Churn']


from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics

forest = RandomForestClassifier()

param_grid = {'criterion':['gini', 'entropy'], 'n_estimators':[1, 10, 100], 'max_depth':[1, 10, 100]}
forest_gscv = GridSearchCV(forest, param_grid, cv=5)
forest_gscv.fit(x_train, y_train)
forest_gscv.predict(x_test)



for i in range(len(forest_gscv.cv_results_["params"])):
    print(forest_gscv.cv_results_["params"][i] , "Accuracy : ", forest_gscv.cv_results_["mean_test_score"][i])

print("\n")
print("Best paramter : ", forest_gscv.best_params_)
print("Best accuracy : ", forest_gscv.best_score_)
print(confusion_matrix(y_test,forest_gscv.predict(x_test)))  
plt.figure(figsize=(2, 2))
sns.heatmap(metrics.confusion_matrix(y_test,forest_gscv.predict(x_test)), annot=True, fmt='.2f', linewidths=.1, cmap='Blues')


nEst_gini = []
mDep_gini = []
value_gini = []
nEst_entro = []
mDep_entro = []
value_entro = []

value = np.array(forest_gscv.cv_results_["mean_test_score"]).reshape(-1, 3)
print(value)

for i in range(len(forest_gscv.cv_results_["params"])):
    param1 = forest_gscv.cv_results_["params"][i]['criterion']
    param2 = forest_gscv.cv_results_["params"][i]['n_estimators']
    param3 = forest_gscv.cv_results_["params"][i]['max_depth']
    
    if(param1 == 'gini'):
        nEst_gini.append(param2)
        mDep_gini.append(param3)
        value_gini.append(forest_gscv.cv_results_["mean_test_score"][i])

    else:
        nEst_entro.append(param2)
        mDep_entro.append(param3)
        value_entro.append(forest_gscv.cv_results_["mean_test_score"][i])