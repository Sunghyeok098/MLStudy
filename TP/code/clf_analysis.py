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
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score

import warnings
warnings.filterwarnings(action='ignore')
pd.set_option('display.max_row', 500)
pd.set_option('display.max_columns', 100)

data_url = 'TP/dataset/classification/labeling_data.csv'
data = pd.read_csv(data_url)

col = [ 'CurrentEquipmentDays', 'MadeCallToRetentionTeam', 'RetentionCalls', 'HandsetWebCapable', 'TotalRecurringCharge', 'MonthlyMinutes', 
'CreditRating', 'OffPeakCallsInOut', 'HandsetModels', 'PeakCallsInOut', 'ReceivedCalls', 'CustomerCareCalls', 'RetentionOffersAccepted',
'UniqueSubs', 'PercChangeMinutes', 'InboundCalls', 'Handsets', 'OutboundCalls', 'Age', 'HandsetRefurbished', 'UnansweredCalls', 'Churn']

select_data = pd.DataFrame()

for i in col:
    select_data = pd.concat([select_data, data[i]], axis=1)

print(select_data)
data_train, data_test = train_test_split(data, test_size=0.2)

x_train = data_train.drop(['Churn'], axis=1)
y_train = data_train['Churn']
x_test = data_test.drop(['Churn'], axis=1)
y_test = data_test['Churn']

def plot_roc_curve(fpr, tpr, label=None):
    plt.plot(fpr,tpr,linewidth=2, label=label)
    plt.plot([0,1],[0,1],'k--')
    plt.axis([0,1,0,1])
    plt.xlabel('FPR',fontsize=16)
    plt.ylabel('TPR',fontsize=16)

"""
# Logistic Regression
print("\nStart LogisticRegression")

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

y_score = logisticRegr_gscv.decision_function(x_train)
fpr, tpr, thresholds = roc_curve(y_train, y_score)

def plot_roc_curve(fpr, tpr, label=None):
    plt.plot(fpr,tpr,linewidth=2, label=label)
    plt.plot([0,1],[0,1],'k--')
    plt.axis([0,1,0,1])
    plt.xlabel('FPR',fontsize=16)
    plt.ylabel('TPR',fontsize=16)

plt.figure(figsize=(8,6))
plot_roc_curve(fpr,tpr,"Logistic Regression")
plt.show()

print('ROC Score : ',roc_auc_score(y_train,y_score))


# KNeighbors Classifier
print("\nStart KNeighborsClassifier")

knnclassifier = KNeighborsClassifier()
param_grid = {'algorithm':['ball_tree', 'kd_tree', 'brute'], 'n_neighbors':[2,3,4,5], 'weights':['uniform', 'distance']}
knnclassifier_gscv = GridSearchCV(knnclassifier, param_grid, cv=5)
knnclassifier_gscv.fit(x_train, y_train)
knnclassifier_gscv.predict(x_test)


for i in range(len(knnclassifier_gscv.cv_results_["params"])):
    print(knnclassifier_gscv.cv_results_["params"][i] , "Accuracy : ", knnclassifier_gscv.cv_results_["mean_test_score"][i])

print("\n")
print("Best parmeter : ", knnclassifier_gscv.best_params_)
print("Best accuracy : ", knnclassifier_gscv.best_score_)    
print(confusion_matrix(y_test, knnclassifier_gscv.predict(x_test)))  
plt.figure(figsize=(2, 2))
sns.heatmap(metrics.confusion_matrix(y_test, knnclassifier_gscv.predict(x_test)), annot=True, fmt='.2f', linewidths=.1, cmap='Blues')

y_score = knnclassifier_gscv.predict_proba(x_train)
fpr_knn, tpr_knn, thresholds_knn = roc_curve(y_train, y_score[:,1])

plt.figure(figsize=(8,6))
plot_roc_curve(fpr_knn,tpr_knn, "KNN")
plt.show()

print('ROC Score : ',roc_auc_score(y_train,y_score[:,1]))


# Decision Tree
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

y_score = deClassifier_gscv.predict_proba(x_train)
fpr_dt, tpr_dt, thresholds_dt = roc_curve(y_train, y_score[:,1])

plt.figure(figsize=(8,6))
plot_roc_curve(fpr_dt,tpr_dt, "Decision Tree")
plt.show()

print('ROC Score : ',roc_auc_score(y_train,y_score[:,1]))


# Random Forest
print("\nStart RandomForest")

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

y_score = forestClassifier_gscv.predict_proba(x_train)
fpr_fc, tpr_fc, thresholds_fc = roc_curve(y_train, y_score[:,1])

plt.figure(figsize=(8,6))
plot_roc_curve(fpr_fc,tpr_fc, "Random Forest")
plt.show()

print('ROC Score : ',roc_auc_score(y_train,y_score[:,1]))
"""




# voting hard
print("\nStart Voting")

log_clf = LogisticRegression(C=0.1, max_iter=100, solver='liblinear')
knn_clf = KNeighborsClassifier(algorithm='ball_tree', n_neighbors=4, weights='uniform')
dec_clf = DecisionTreeClassifier(criterion='entropy', max_depth=5)
rnd_clf = RandomForestClassifier(criterion='entropy', max_depth=100, n_estimators=100)

voting_clf = VotingClassifier(estimators=[('lr', log_clf)], voting='hard')
voting_clf.fit(x_train, y_train)
y_pred = voting_clf.predict(x_test)
print(accuracy_score(y_test,y_pred))
 
voting_clf = VotingClassifier(estimators=[('kn', knn_clf)], voting='hard')
voting_clf.fit(x_train, y_train)
y_pred = voting_clf.predict(x_test)
print(accuracy_score(y_test,y_pred))

voting_clf = VotingClassifier(estimators=[('dc', dec_clf)], voting='hard')
voting_clf.fit(x_train, y_train)
y_pred = voting_clf.predict(x_test)
print(accuracy_score(y_test,y_pred))

voting_clf = VotingClassifier(estimators=[('rf', rnd_clf)], voting='hard')
voting_clf.fit(x_train, y_train)
y_pred = voting_clf.predict(x_test)
print(accuracy_score(y_test,y_pred))



voting_clf = VotingClassifier(estimators=[('lr', log_clf)], voting='soft')
voting_clf.fit(x_train, y_train)
y_pred = voting_clf.predict(x_test)
print(accuracy_score(y_test,y_pred))
 
voting_clf = VotingClassifier(estimators=[('kn', knn_clf)], voting='soft')
voting_clf.fit(x_train, y_train)
y_pred = voting_clf.predict(x_test)
print(accuracy_score(y_test,y_pred))

voting_clf = VotingClassifier(estimators=[('dc', dec_clf)], voting='soft')
voting_clf.fit(x_train, y_train)
y_pred = voting_clf.predict(x_test)
print(accuracy_score(y_test,y_pred))

voting_clf = VotingClassifier(estimators=[('rf', rnd_clf)], voting='soft')
voting_clf.fit(x_train, y_train)
y_pred = voting_clf.predict(x_test)
print(accuracy_score(y_test,y_pred))




"""
# SVC
print("\nStart SVC")

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

y_score = svclassifier_gscv.decision_function(x_train)
fpr_svc, tpr_svc, thresholds_svc = roc_curve(y_train, y_score)

plt.figure(figsize=(8,6))
plot_roc_curve(fpr_svc,tpr_svc, "SVC")
plt.show()

print('ROC Score : ',roc_auc_score(y_train,y_score))
"""