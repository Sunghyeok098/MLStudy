# import libraries
from sklearn.svm import SVC
from sklearn.metrics import classification_report,  confusion_matrix
from sklearn.model_selection import train_test_split, GridSearchCV
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import axes3d
from matplotlib import style
style.use('ggplot')

data_url = 'Ex\dataSet\week3_dataSet\car_mini.csv'
dataSet = pd.read_csv(data_url, names=['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'car'])

data = dataSet.copy()


data.car = data.car.replace('vgood', 'acc')
data.car = data.car.replace('good', 'acc')

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
data['buying'] = le.fit_transform(data['buying'])
data['maint'] = le.fit_transform(data['maint'])
data['doors'] = le.fit_transform(data['doors'])
data['persons'] = le.fit_transform(data['persons'])
data['lug_boot'] = le.fit_transform(data['lug_boot'])
data['safety'] = le.fit_transform(data['safety'])
data['car'] = le.fit_transform(data['car'])

#print(data)

print("\nStart RandomForest")

data_train, data_test = train_test_split(data, test_size=0.2)

x_train = data_train.drop(['car'], axis=1)
y_train = data_train['car']
x_test = data_test.drop(['car'], axis=1)
y_test = data_test['car']


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

list1 = []
list2 = []
list3 = []

for i in range(len(forest_gscv.cv_results_["params"])):
    param1 = forest_gscv.cv_results_["params"][i]['criterion']
    param2 = forest_gscv.cv_results_["params"][i]['n_estimators']
    param3 = forest_gscv.cv_results_["params"][i]['max_depth']
    
    if(param1 == 'gini'):
        list1.append(0)
    else:
        list1.append(1)
    list2.append(param2)
    list3.append(param3)

    
value = np.array(forest_gscv.cv_results_["mean_test_score"]).tolist()

fig = plt.figure()
ax1 = fig.add_subplot(111, projection='3d')

"""
xpos = [1, 10, 100]
ypos = [1, 10, 100]
num_elements = len(xpos)
zpos = [0,0,0,0,0,0,0,0,0,0]
dx = list2
dy = list3
dz = value
ax1.bar3d(xpos, ypos, zpos, dx, dy, dz, color='#00ceaa')

ax1.set_xlabel('x axis')
ax1.set_ylabel('y axis')
ax1.set_zlabel('z axis')

plt.show()
"""


print("\nStart LogisticRegression")
from sklearn.linear_model import LogisticRegression

logisticRegr = LogisticRegression(C=0.1, solver="liblinear", max_iter=50)
param_grid = {'C':[0.1, 1, 10], 'solver':['liblinear', 'lbfgs', 'sag'], 'max_iter':[50, 100, 200]}
logisticRegr_gscv = GridSearchCV(logisticRegr, param_grid, cv=5)
logisticRegr_gscv.fit(x_train, y_train)
logisticRegr_gscv.predict(x_test)


for i in range(len(logisticRegr_gscv.cv_results_["params"])):
    print(logisticRegr_gscv.cv_results_["params"][i] , "Accuracy : ", logisticRegr_gscv.cv_results_["mean_test_score"][i])

print("\n")
print("Best parameter : ",logisticRegr_gscv.best_params_)
print("Best accuracy : ", logisticRegr_gscv.best_score_)
print(confusion_matrix(y_test,logisticRegr_gscv.predict(x_test))) 
plt.figure(figsize=(2, 2))
sns.heatmap(metrics.confusion_matrix(y_test,logisticRegr_gscv.predict(x_test)), annot=True, fmt='.2f', linewidths=.1, cmap='Blues')







print("\nStart SVM")
from sklearn.svm import SVC

svclassifier = SVC(kernel='linear' )
param_grid = {'C':[0.1, 1, 10], 'kernel':['linear', 'poly', 'rbf', 'sigmoid'], 'gamma':[0.01, 0.1, 1.0, 10.0]}
svclassifier_gscv = GridSearchCV(svclassifier, param_grid, cv=5)
svclassifier_gscv.fit(x_train, y_train)
svclassifier_gscv.predict(x_test)


for i in range(len(svclassifier_gscv.cv_results_["params"])):
    print(svclassifier_gscv.cv_results_["params"][i] , "Accuracy : ", svclassifier_gscv.cv_results_["mean_test_score"][i])

print("\n")
print("Best parmeter : ",svclassifier_gscv.best_params_)
print("Best accuracy : ", svclassifier_gscv.best_score_)    
print(confusion_matrix(y_test,svclassifier_gscv.predict(x_test)))  
plt.figure(figsize=(2, 2))
sns.heatmap(metrics.confusion_matrix(y_test,svclassifier_gscv.predict(x_test)), annot=True, fmt='.2f', linewidths=.1, cmap='Blues')
plt.show()



