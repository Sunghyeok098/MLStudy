import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings(action='ignore')
pd.set_option('display.max_row', 500)
pd.set_option('display.max_columns', 100)

data_url = 'TP/dataset/classification/cell2celltrain.csv'
data = pd.read_csv(data_url)

#print(data.head(5))
#print(data.describe())
#print(data.info())
#print(data.isna().sum())


data = data.drop(['CustomerID'], axis=1)
data = data.drop(['ServiceArea'], axis=1)

data.CreditRating = data.CreditRating.replace('1-Highest', 'Highest')
data.CreditRating = data.CreditRating.replace('2-High', 'High')
data.CreditRating = data.CreditRating.replace('3-Good', 'Good')
data.CreditRating = data.CreditRating.replace('4-Medium', 'Medium')
data.CreditRating = data.CreditRating.replace('5-Low', 'Low')
data.CreditRating = data.CreditRating.replace('6-VeryLow', 'VeryLow')
data.CreditRating = data.CreditRating.replace('7-Lowest', 'Lowest')

#print(data.isna().sum())
#print(data)

df = pd.DataFrame()

from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn import linear_model

le = LabelEncoder()
data['Churn'] = le.fit_transform(data['Churn'])
data['MaritalStatus'] = le.fit_transform(data['MaritalStatus'])



def reg_columns(df, predictor, predicted, round_num):

    x = data[predictor]
    y = data[predicted]

    x_train, y_train, x_test = [], [], []

    for i in range(len(y)):
        if (np.isnan(y[i])):
            x_test.append(x[i])
        else:
            x_train.append(x[i])
            y_train.append(y[i])


    E = linear_model.LinearRegression()
    E.fit(np.array(x_train)[:, np.newaxis], np.array(y_train))
    y_pred = E.predict(np.array(x_test)[:, np.newaxis])

    j=0

    for i in range(len(y)):
        if (np.isnan(y[i])):
            y[i] = round(y_pred[j], round_num)
            j+=1
    
reg_columns(df, 'Churn', 'MonthlyRevenue', 2)
reg_columns(df, 'Churn', 'MonthlyMinutes', 0)
reg_columns(df, 'Churn', 'TotalRecurringCharge',0)
reg_columns(df, 'Churn', 'DirectorAssistedCalls', 2)
reg_columns(df, 'Churn', 'OverageMinutes', 0)
reg_columns(df, 'Churn', 'RoamingCalls', 1)
reg_columns(df, 'Churn', 'PercChangeMinutes', 0)
reg_columns(df, 'Churn', 'PercChangeRevenues', 1)
reg_columns(df, 'Churn', 'Handsets', 0)
reg_columns(df, 'Churn', 'HandsetPrice', 0)
reg_columns(df, 'Churn', 'HandsetModels', 0)
reg_columns(df, 'Churn', 'CurrentEquipmentDays', 0)
reg_columns(df, 'Churn', 'AgeHH1', 0)


from sklearn.linear_model import  LogisticRegression

def logisticReg_columns(df, predictor, predicted):
    
    x = data[predictor]
    y = data[predicted]

    x_train, y_train, x_test = [], [], []

    for i in range(len(y)):
        if (y[i] == 2):
            x_test.append(x[i])
        else:
            x_train.append(x[i])
            y_train.append(y[i])


    logisticRegr = LogisticRegression()
    logisticRegr.fit(np.array(x_train).reshape(-1,1), np.array(y_train).reshape(-1,1))
    y_pred = logisticRegr.predict(np.array(x_test).reshape(-1,1))

    j=0

    for i in range(len(y)):
        if (y[i] == 2 and y_pred[j] == 0):
            y[i] = 'No'
            j+=1

        elif (y[i] == 2 and y_pred[j] == 1):
            y[i] = 'Yes'
            j+=1


logisticReg_columns(df, 'Churn', 'MaritalStatus')


for i in data.columns:
    if(data[i].isna().sum() == 0):
        df = pd.concat([df, data[i]], axis=1)

df.rename(columns = {'AgeHH1': 'Age'}, inplace = True)
df.Churn = df.Churn.replace(1, 'Yes')
df.Churn = df.Churn.replace(0, 'No')
df.MaritalStatus = df.MaritalStatus.replace(1, 'Yes')
df.MaritalStatus = df.MaritalStatus.replace(0, 'No')

print(df.info())
print(df.isna().sum())
print(df)



df.to_csv('TP/dataset/classification/preprocessing_data.csv', index=False, encoding='cp949')
