# import libraries
from sklearn.svm import SVC
from sklearn.metrics import classification_report,  confusion_matrix
from sklearn.model_selection import train_test_split
import pandas as pd

data_url = 'Ex/dataSet/week2_dataSet/bill_authentication.csv'
dataSet = pd.read_csv(data_url)

data = dataSet.copy()

X = data.drop(['Class'], axis=1)
y = data['Class']

# split the dataset into train set and test set  
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20)

# create an SVC classifier model 
svclassifier = SVC(kernel='linear' )

#svclassifier = SVC(kernel='sigmoid',C=1.0, degree=3,  gamma='auto', coef0=0.0, shrinking=True,  probability=False,tol=0.001, cache_size=200,  class_weight=None, verbose=False, max_iter=-1,  random_state=None)

# fit the model to train dataset  
svclassifier.fit(X_train, y_train)

# make predictions using the trained model  
y_pred = svclassifier.predict(X_test)

# evaluate the model  
print(confusion_matrix(y_test,y_pred))  
print(classification_report(y_test,y_pred))
