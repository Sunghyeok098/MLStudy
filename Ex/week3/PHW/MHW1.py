# import libraries
from sklearn.svm import SVC
from sklearn.metrics import classification_report,  confusion_matrix
from sklearn.model_selection import train_test_split, GridSearchCV
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import metrics

from mpl_toolkits.mplot3d import axes3d
from matplotlib import style
style.use('ggplot')

data_url = 'Ex\dataSet\week3_dataSet\mnist.csv'
dataSet = pd.read_csv(data_url)

data = dataSet.copy()
#data = data[data['label'].isin(['0', '1'])]

a, d1 = train_test_split(data, test_size=0.1)
a, d2 = train_test_split(data, test_size=0.1)
train_d1, test_d1 = train_test_split(d1, test_size=0.1)

x_train = train_d1.drop(['label'], axis=1)
y_train = train_d1['label']
x_test = test_d1.drop(['label'], axis=1)
y_test = test_d1['label']

d2_x = d2.drop(['label'], axis=1)
d2_y = d2['label']



print("\nStart RandomForest")
from sklearn.ensemble import RandomForestClassifier

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
forest_accuracy = forest_gscv.best_score_
print(confusion_matrix(y_test,forest_gscv.predict(x_test)))  
plt.figure(figsize=(6, 6))

sns.heatmap(metrics.confusion_matrix(y_test,forest_gscv.predict(x_test)), annot=True, fmt='.2f', linewidths=.1, cmap='Blues')
plt.title('Random Forest')


value_gini = []
value_entro = []

value = np.array(forest_gscv.cv_results_["mean_test_score"]).reshape(-1, 3)

for i in range(len(forest_gscv.cv_results_["params"])):
    param1 = forest_gscv.cv_results_["params"][i]['criterion']

    if(param1 == 'gini'):
        value_gini.append(forest_gscv.cv_results_["mean_test_score"][i])

    else:
        value_entro.append(forest_gscv.cv_results_["mean_test_score"][i])



fig = plt.figure(figsize=(8, 8))
ax1 = fig.add_subplot(projection='3d')

xlabels = np.array(['1', '10', '100',])
xpos = np.arange(xlabels.shape[0])
ylabels = np.array(['1', '10', '100',])
ypos = np.arange(ylabels.shape[0])
xposM, yposM = np.meshgrid(xpos, ypos, copy=False)

zpos=value[0:3]
zpos = zpos.ravel()


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


value_entro = np.array(value_entro).reshape(-1,3)

fig = plt.figure(figsize=(8, 8))
ax1 = fig.add_subplot(projection='3d')

xlabels = np.array(['1', '10', '100',])
xpos = np.arange(xlabels.shape[0])
ylabels = np.array(['1', '10', '100',])
ypos = np.arange(ylabels.shape[0])
xposM, yposM = np.meshgrid(xpos, ypos, copy=False)


zpos= value_entro.flatten()


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
logistic_accuracy = logisticRegr_gscv.best_score_
print(confusion_matrix(y_test,logisticRegr_gscv.predict(x_test))) 
plt.figure(figsize=(6, 6))
plt.title('logisticRegression')
sns.heatmap(metrics.confusion_matrix(y_test,logisticRegr_gscv.predict(x_test)), annot=True, fmt='.2f', linewidths=.1, cmap='Blues')


value_liblinear = []
value_lbfgs = []
value_sag = []

value = np.array(logisticRegr_gscv.cv_results_["mean_test_score"]).reshape(-1, 3)


for i in range(len(logisticRegr_gscv.cv_results_["params"])):
    param1 = logisticRegr_gscv.cv_results_["params"][i]['solver']
    
    if(param1 == 'liblinear'):
        value_liblinear.append(logisticRegr_gscv.cv_results_["mean_test_score"][i])
    
    elif(param1 == 'lbfgs'):
        value_lbfgs.append(logisticRegr_gscv.cv_results_["mean_test_score"][i])

    else:
        value_sag.append(logisticRegr_gscv.cv_results_["mean_test_score"][i])


value_liblinear = np.array(value_liblinear).reshape(-1,3)
fig = plt.figure(figsize=(8, 8))
ax1 = fig.add_subplot(projection='3d')

xlabels = np.array(['1', '10', '100',])
xpos = np.arange(xlabels.shape[0])
ylabels = np.array(['1', '10', '100',])
ypos = np.arange(ylabels.shape[0])
xposM, yposM = np.meshgrid(xpos, ypos, copy=False)

z= value_liblinear.flatten()

dx=0.5
dy=0.5
dz=z
ax1.w_xaxis.set_ticks(xpos + dx/2.)
ax1.w_xaxis.set_ticklabels(xlabels)

ax1.w_yaxis.set_ticks(ypos + dy/2.)
ax1.w_yaxis.set_ticklabels(ylabels)
ax1.bar3d(xposM.ravel(), yposM.ravel(), dz*0, dx, dy, dz)


ax1.set_xlabel('C')
ax1.set_ylabel('max_iter')
ax1.set_zlabel('Accuracy')
ax1.set_title('liblinear')



value_lbfgs = np.array(value_lbfgs).reshape(-1,3)
fig = plt.figure(figsize=(8, 8))
ax1 = fig.add_subplot(projection='3d')

xlabels = np.array(['1', '10', '100',])
xpos = np.arange(xlabels.shape[0])
ylabels = np.array(['1', '10', '100',])
ypos = np.arange(ylabels.shape[0])
xposM, yposM = np.meshgrid(xpos, ypos, copy=False)

z= value_lbfgs.flatten()

dx=0.5
dy=0.5
dz=z
ax1.w_xaxis.set_ticks(xpos + dx/2.)
ax1.w_xaxis.set_ticklabels(xlabels)

ax1.w_yaxis.set_ticks(ypos + dy/2.)
ax1.w_yaxis.set_ticklabels(ylabels)
ax1.bar3d(xposM.ravel(), yposM.ravel(), dz*0, dx, dy, dz)

ax1.set_xlabel('C')
ax1.set_ylabel('max_iter')
ax1.set_zlabel('Accuracy')
ax1.set_title('lbfgs')


value_sag = np.array(value_sag).reshape(-1,3)
fig = plt.figure(figsize=(8, 8))
ax1 = fig.add_subplot(projection='3d')

xlabels = np.array(['1', '10', '100',])
xpos = np.arange(xlabels.shape[0])
ylabels = np.array(['1', '10', '100',])
ypos = np.arange(ylabels.shape[0])
xposM, yposM = np.meshgrid(xpos, ypos, copy=False)

z= value_sag.flatten()

dx=0.5
dy=0.5
dz=z
ax1.w_xaxis.set_ticks(xpos + dx/2.)
ax1.w_xaxis.set_ticklabels(xlabels)

ax1.w_yaxis.set_ticks(ypos + dy/2.)
ax1.w_yaxis.set_ticklabels(ylabels)
ax1.bar3d(xposM.ravel(), yposM.ravel(), dz*0, dx, dy, dz)

ax1.set_xlabel('C')
ax1.set_ylabel('max_iter')
ax1.set_zlabel('Accuracy')
ax1.set_title('logistic - sag')

plt.show()











print("\nStart SVM")
from sklearn.svm import SVC

svclassifier = SVC()
param_grid = {'C':[0.1, 1, 10], 'kernel':['linear', 'poly', 'rbf', 'sigmoid'], 'gamma':[0.01, 0.1, 1.0, 10.0]}
svclassifier_gscv = GridSearchCV(svclassifier, param_grid, cv=5)
svclassifier_gscv.fit(x_train, y_train)
svclassifier_gscv.predict(x_test)


for i in range(len(svclassifier_gscv.cv_results_["params"])):
    print(svclassifier_gscv.cv_results_["params"][i] , "Accuracy : ", svclassifier_gscv.cv_results_["mean_test_score"][i])

print("\n")
print("Best parmeter : ",svclassifier_gscv.best_params_)
print("Best accuracy : ", svclassifier_gscv.best_score_)    
svm_accuracy = svclassifier_gscv.best_score_
print(confusion_matrix(y_test,svclassifier_gscv.predict(x_test)))  
plt.figure(figsize=(6, 6))
plt.title('svm')
sns.heatmap(metrics.confusion_matrix(y_test,svclassifier_gscv.predict(x_test)), annot=True, fmt='.2f', linewidths=.1, cmap='Blues')




value_linear = []
value_poly = []
value_rbf = []
value_sigmoid = []


for i in range(len(svclassifier_gscv.cv_results_["params"])):
    param1 = svclassifier_gscv.cv_results_["params"][i]['kernel']
    
    if(param1 == 'linear'):
        value_linear.append(svclassifier_gscv.cv_results_["mean_test_score"][i])
    
    elif(param1 == 'poly'):
        value_poly.append(svclassifier_gscv.cv_results_["mean_test_score"][i])
    
    elif(param1 == 'rbf'):
        value_rbf.append(svclassifier_gscv.cv_results_["mean_test_score"][i])

    else:
        value_sigmoid.append(svclassifier_gscv.cv_results_["mean_test_score"][i])


value_linear = np.array(value_linear).reshape(-1,4)
fig = plt.figure(figsize=(8, 8))
ax1 = fig.add_subplot(projection='3d')

xlabels = np.array(['0.1', '1', '10'])
xpos = np.arange(xlabels.shape[0])
ylabels = np.array(['0.01', '0.1', '1.0', '10.0'])
ypos = np.arange(ylabels.shape[0])
xposM, yposM = np.meshgrid(xpos, ypos, copy=False)


z= value_linear.flatten()

dx=0.5
dy=0.5
dz=z
ax1.w_xaxis.set_ticks(xpos + dx/2.)
ax1.w_xaxis.set_ticklabels(xlabels)

ax1.w_yaxis.set_ticks(ypos + dy/2.)
ax1.w_yaxis.set_ticklabels(ylabels)
ax1.bar3d(xposM.ravel(), yposM.ravel(), dz*0, dx, dy, dz)


ax1.set_xlabel('C')
ax1.set_ylabel('gamma')
ax1.set_zlabel('Accuracy')
ax1.set_title('svm - linear')



value_poly = np.array(value_poly).reshape(-1,4)
fig = plt.figure(figsize=(8, 8))
ax1 = fig.add_subplot(projection='3d')

xlabels = np.array(['0.1', '1', '10'])
xpos = np.arange(xlabels.shape[0])
ylabels = np.array(['0.01', '0.1', '1', '10'])
ypos = np.arange(ylabels.shape[0])
xposM, yposM = np.meshgrid(xpos, ypos, copy=False)

z= value_poly.flatten()

dx=0.5
dy=0.5
dz=z
ax1.w_xaxis.set_ticks(xpos + dx/2.)
ax1.w_xaxis.set_ticklabels(xlabels)

ax1.w_yaxis.set_ticks(ypos + dy/2.)
ax1.w_yaxis.set_ticklabels(ylabels)
ax1.bar3d(xposM.ravel(), yposM.ravel(), dz*0, dx, dy, dz)

ax1.set_xlabel('C')
ax1.set_ylabel('gamma')
ax1.set_zlabel('Accuracy')
ax1.set_title('svm - poly')


value_rbf = np.array(value_rbf).reshape(-1,4)
fig = plt.figure(figsize=(8, 8))
ax1 = fig.add_subplot(projection='3d')

xlabels = np.array(['0.1', '1', '10'])
xpos = np.arange(xlabels.shape[0])
ylabels = np.array(['0.01', '0.1', '1.0', '10.0'])
ypos = np.arange(ylabels.shape[0])
xposM, yposM = np.meshgrid(xpos, ypos, copy=False)

z= value_rbf.flatten()

dx=0.5
dy=0.5
dz=z
ax1.w_xaxis.set_ticks(xpos + dx/2.)
ax1.w_xaxis.set_ticklabels(xlabels)

ax1.w_yaxis.set_ticks(ypos + dy/2.)
ax1.w_yaxis.set_ticklabels(ylabels)
ax1.bar3d(xposM.ravel(), yposM.ravel(), dz*0, dx, dy, dz)

ax1.set_xlabel('C')
ax1.set_ylabel('gamma')
ax1.set_zlabel('Accuracy')
ax1.set_title('svm - rbf')

value_sigmoid = np.array(value_sigmoid).reshape(-1,4)
fig = plt.figure(figsize=(8, 8))
ax1 = fig.add_subplot(projection='3d')

xlabels = np.array(['0.1', '1', '10'])
xpos = np.arange(xlabels.shape[0])
ylabels = np.array(['0.01', '0.1', '1.0', '10.0'])
ypos = np.arange(ylabels.shape[0])
xposM, yposM = np.meshgrid(xpos, ypos, copy=False)

z= value_sigmoid.flatten()

dx=0.5
dy=0.5
dz=z
ax1.w_xaxis.set_ticks(xpos + dx/2.)
ax1.w_xaxis.set_ticklabels(xlabels)

ax1.w_yaxis.set_ticks(ypos + dy/2.)
ax1.w_yaxis.set_ticklabels(ylabels)
ax1.bar3d(xposM.ravel(), yposM.ravel(), dz*0, dx, dy, dz)

ax1.set_xlabel('C')
ax1.set_ylabel('gamma')
ax1.set_zlabel('Accuracy')
ax1.set_title('svm - sigmoid')


plt.show()


print("\nEnsemble classifier start")
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score

vote_clf = VotingClassifier(estimators=[('random_forest', forest_gscv), ('logistic_regression', logisticRegr_gscv), ('svc', svclassifier_gscv)], voting='hard')

vote_clf = vote_clf.fit(x_train, y_train)
vote_predict = vote_clf.predict(d2_x)

print("Ensemble classifier accuracy : ", accuracy_score(d2_y, vote_predict))
ensemble_accuracy = accuracy_score(d2_y, vote_predict)
#Get the confusion matrix
cf_matrix = confusion_matrix(d2_y, vote_predict)
print(cf_matrix)

plt.figure(figsize=(6, 6))
sns.heatmap(metrics.confusion_matrix(d2_y, vote_predict), annot=True, fmt='.2f', linewidths=.1, cmap='Blues')
plt.title("Ensemble classifier Confusion Matrix")
plt.show()


print("Random forest accuracy : ", forest_accuracy)
print("logistic regreesion accuracy : ", logistic_accuracy)
print("svm accuracy : ", svm_accuracy)
print("ensemble accurcay : ", ensemble_accuracy)