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

print(nEst_gini)
print(mDep_gini)
#print(value_gini)

fig = plt.figure(figsize=(8, 8))
ax1 = fig.add_subplot(projection='3d')

xlabels = np.array(['1', '10', '100',])
xpos = np.arange(xlabels.shape[0])
ylabels = np.array(['1', '10', '100',])
ypos = np.arange(ylabels.shape[0])
xposM, yposM = np.meshgrid(xpos, ypos, copy=False)

zpos=value[0:3]
print(zpos)
zpos = zpos.ravel()
print(type(zpos))
print(zpos)

dx=0.5
dy=0.5
dz=zpos
ax1.w_xaxis.set_ticks(xpos + dx/2.)
ax1.w_xaxis.set_ticklabels(xlabels)

ax1.w_yaxis.set_ticks(ypos + dy/2.)
ax1.w_yaxis.set_ticklabels(ylabels)
ax1.bar3d(xposM.ravel(), yposM.ravel(), dz*0, dx, dy, dz)


ax1.set_xlabel('n_estimators')
ax1.set_ylabel('max_depth')
ax1.set_zlabel('Accuracy')
ax1.set_title('Gini')

plt.show()


print(nEst_entro)
print(mDep_entro)
print(value_entro)

value_entro = np.array(value_entro).reshape(-1,3)
print(value_entro)
fig = plt.figure(figsize=(8, 8))
ax1 = fig.add_subplot(projection='3d')

xlabels = np.array(['1', '10', '100',])
xpos = np.arange(xlabels.shape[0])
ylabels = np.array(['1', '10', '100',])
ypos = np.arange(ylabels.shape[0])
xposM, yposM = np.meshgrid(xpos, ypos, copy=False)

print(value_entro)
z= value_entro.flatten()
print(value_entro)
print(type(z))

dx=0.5
dy=0.5
dz=zpos
ax1.w_xaxis.set_ticks(xpos + dx/2.)
ax1.w_xaxis.set_ticklabels(xlabels)

ax1.w_yaxis.set_ticks(ypos + dy/2.)
ax1.w_yaxis.set_ticklabels(ylabels)
ax1.bar3d(xposM.ravel(), yposM.ravel(), dz*0, dx, dy, dz)

ax1.set_xlabel('n_estimators')
ax1.set_ylabel('max_depth')
ax1.set_zlabel('Accuracy')
ax1.set_title('Entropy')

plt.show()



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
#plt.show()

