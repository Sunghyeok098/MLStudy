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
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif

import warnings
warnings.filterwarnings(action='ignore')
pd.set_option('display.max_row', 500)
pd.set_option('display.max_columns', 100)

data_url = 'TP/dataset/classification/labeling_data2.csv'
data = pd.read_csv(data_url)

#print(data)

x = pd.get_dummies(data.drop(['Churn'], axis=1))
y = data['Churn']

bestfeature = SelectKBest(f_classif, k='all')
fit = bestfeature.fit(x, y)

dfcolumns = pd.DataFrame(x.columns)
dfscores = pd.DataFrame(fit.scores_)

featureScores = pd.concat([dfcolumns, dfscores], axis=1)
featureScores.columns = ['feature', 'Score']

print(featureScores.nlargest(60, 'Score'))

# correlation hour-per-week with other feature
corrmat = data.corr()  # corr() computes pairwise  correlations of features in a Data Frame
top_corr_features = corrmat.index
plt.figure(figsize=(20, 20)).show()
# plot the heat map
sns.heatmap(data[top_corr_features].corr(), annot=True, cmap="RdYlGn")
plt.show()