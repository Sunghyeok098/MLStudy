import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.metrics import classification_report,  confusion_matrix
from sklearn.model_selection import train_test_split, GridSearchCV
from mpl_toolkits.mplot3d import axes3d
from matplotlib import style
from sklearn import metrics

import warnings
warnings.filterwarnings(action='ignore')
pd.set_option('display.max_row', 500)
pd.set_option('display.max_columns', 100)

data_url = 'TP/dataset/classification/labeling_data.csv'
data = pd.read_csv(data_url)

data_train, data_test = train_test_split(data, test_size=0.2)

x_train = data_train.drop(['Churn'], axis=1)
y_train = data_train['Churn']
x_test = data_test.drop(['Churn'], axis=1)
y_test = data_test['Churn']


print("\nStart LogisticRegression")
from sklearn.linear_model import LogisticRegression

logisticRegr = LogisticRegression()
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

svclassifier = SVC()
param_grid = {'C':[0.1, 1, 10], 'kernel':['linear', 'rbf', 'sigmoid'], 'gamma':[0.01, 0.1, 1.0, 10.0]}
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


from sklearn.neighbors import KNeighborsClassifier

knnclassifier = KNeighborsClassifier()
param_grid = {'algorithm':['ball_tree', 'kd_tree', 'brute'], 'n_neighbors':[2,3,4,5], 'weights':['uniform', 'distance']}
knnclassifier_gscv = GridSearchCV(knnclassifier, param_grid, cv=5)
knnclassifier_gscv.fit(x_train, y_train)
knnclassifier_gscv.predict(x_test)


for i in range(len(svclassifier_gscv.cv_results_["params"])):
    print(svclassifier_gscv.cv_results_["params"][i] , "Accuracy : ", svclassifier_gscv.cv_results_["mean_test_score"][i])

print("\n")
print("Best parmeter : ", knnclassifier_gscv.best_params_)
print("Best accuracy : ", knnclassifier_gscv.best_score_)    
print(confusion_matrix(y_test, knnclassifier_gscv.predict(x_test)))  
plt.figure(figsize=(2, 2))
sns.heatmap(metrics.confusion_matrix(y_test, knnclassifier_gscv.predict(x_test)), annot=True, fmt='.2f', linewidths=.1, cmap='Blues')



print("\nStart DecisionTree")
from sklearn.tree import DecisionTreeClassifier

deClassifier = DecisionTreeClassifier()
param_grid = {'criterion':['gini', 'entropy'], 'max_depth':[5, 10, 15, 20]}
deClassifier_gscv = GridSearchCV(deClassifier, param_grid, cv=5)
deClassifier_gscv.fit(x_train, y_train)
deClassifier_gscv.predict(x_test)


for i in range(len(deClassifier_gscv.cv_results_["params"])):
    print(deClassifier_gscv.cv_results_["params"][i] , "Accuracy : ", deClassifier_gscv.cv_results_["mean_test_score"][i])

print("\n")
print("Best parmeter : ", deClassifier_gscv.best_params_)
print("Best accuracy : ", deClassifier_gscv.best_score_)    
print(confusion_matrix(y_test, deClassifier_gscv.predict(x_test)))  
plt.figure(figsize=(2, 2))
sns.heatmap(metrics.confusion_matrix(y_test, deClassifier_gscv.predict(x_test)), annot=True, fmt='.2f', linewidths=.1, cmap='Blues')


print("\nStart RandomForest")

from sklearn.ensemble import RandomForestClassifier

forestClassifier = RandomForestClassifier()

param_grid = {'criterion':['gini', 'entropy'], 'n_estimators':[1, 10, 100], 'max_depth':[1, 10, 100]}
forestClassifier_gscv = GridSearchCV(forestClassifier, param_grid, cv=5)
forestClassifier_gscv.fit(x_train, y_train)
forestClassifier_gscv.predict(x_test)

for i in range(len(forestClassifier_gscv.cv_results_["params"])):
    print(forestClassifier_gscv.cv_results_["params"][i] , "Accuracy : ", forestClassifier_gscv.cv_results_["mean_test_score"][i])

print("\n")
print("Best paramter : ", forestClassifier_gscv.best_params_)
print("Best accuracy : ", forestClassifier_gscv.best_score_)
print(confusion_matrix(y_test,forestClassifier_gscv.predict(x_test)))  
plt.figure(figsize=(2, 2))
sns.heatmap(metrics.confusion_matrix(y_test,forestClassifier_gscv.predict(x_test)), annot=True, fmt='.2f', linewidths=.1, cmap='Blues')


from sklearn.ensemble import VotingClassifier

voteClassifier = VotingClassifier()

param_grid = {'voting':['hard', 'soft']}
voteClassifier_gscv = GridSearchCV(voteClassifier, param_grid, cv=5)
voteClassifier_gscv.fit(x_train, y_train)
voteClassifier_gscv.predict(x_test)

for i in range(len(voteClassifier_gscv.cv_results_["params"])):
    print(voteClassifier_gscv.cv_results_["params"][i] , "Accuracy : ", voteClassifier_gscv.cv_results_["mean_test_score"][i])

print("\n")
print("Best paramter : ", voteClassifier_gscv.best_params_)
print("Best accuracy : ", voteClassifier_gscv.best_score_)
print(confusion_matrix(y_test, voteClassifier_gscv.predict(x_test)))  
plt.figure(figsize=(2, 2))
sns.heatmap(metrics.confusion_matrix(y_test, voteClassifier_gscv.predict(x_test)), annot=True, fmt='.2f', linewidths=.1, cmap='Blues')