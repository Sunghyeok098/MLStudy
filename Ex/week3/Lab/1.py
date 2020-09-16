# import libraries
from sklearn.svm import SVC
from sklearn.metrics import classification_report,  confusion_matrix
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

data_url = 'Ex\dataSet\week3_dataSet\car_mini.csv'
dataSet = pd.read_csv(data_url, names=['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'car'])

data = dataSet.copy()

#print(data.columns)
#print(data)
#print(data['car'].unique())

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

from sklearn.model_selection import KFold

kfold = KFold(10, True, 1)

print("\nStart RandomForest")
sum = 0

for train, test in kfold.split(data):

    data_train = data.loc[[i for i in train], :]
    data_test = data.loc[[i for i in test], :]

    x_train = data_train.drop(['car'], axis=1)
    y_train = data_train['car']
    x_test = data_test.drop(['car'], axis=1)
    y_test = data_test['car']

    from sklearn.ensemble import RandomForestClassifier
    from sklearn import metrics

    forest = RandomForestClassifier(n_estimators=100)
    forest.fit(x_train, y_train)

    y_pred = forest.predict(x_test)
    score = forest.score(x_test, y_test)

    sum = sum + score

    #print(confusion_matrix(y_test,y_pred))  
    #print(classification_report(y_test,y_pred))
    #print('Random Forest Predict score :', metrics.accuracy_score(y_test, y_pred))
    

print("Kfold = 10 RandomForest Average accuracy : ", sum/10)

sum = 0;
print("\nStart logisticRegression")
count = 0
for train, test in kfold.split(data):

    data_train = data.loc[[i for i in train], :]
    data_test = data.loc[[i for i in test], :]

    x_train = data_train.drop(['car'], axis=1)
    y_train = data_train['car']
    x_test = data_test.drop(['car'], axis=1)
    y_test = data_test['car']

    from sklearn.linear_model import LogisticRegression

    logisticRegr = LogisticRegression(C=0.1, solver="liblinear", max_iter=50)
    logisticRegr.fit(x_train, y_train)

    y_pred = logisticRegr.predict(x_test)

    score = logisticRegr.score(x_test, y_test)  
    sum = sum + score

    #print(metrics.confusion_matrix(y_test, y_pred)) 
    #print(classification_report(y_test, y_pred))

   
   
print("Kfold = 10 logisticRegression Average accuracy : ", sum/10)

sum = 0;
print("\nStart SVM")
for train, test in kfold.split(data):

    data_train = data.loc[[i for i in train], :]
    data_test = data.loc[[i for i in test], :]

    x_train = data_train.drop(['car'], axis=1)
    y_train = data_train['car']
    x_test = data_test.drop(['car'], axis=1)
    y_test = data_test['car']

    from sklearn.svm import SVC

    svclassifier = SVC(kernel='linear' )
    svclassifier.fit(x_train, y_train)
    y_pred = svclassifier.predict(x_test)

    score = svclassifier.score(x_test, y_test)  
    sum = sum + score

    #print(confusion_matrix(y_test,y_pred))  
    #print(classification_report(y_test,y_pred))


print("Kfold = 10 SVM Average accuracy : ", sum/10)

