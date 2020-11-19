import pandas as pd
from pandas import DataFrame
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

import warnings
warnings.filterwarnings(action='ignore')
pd.set_option('display.max_row', 500)
pd.set_option('display.max_columns', 100)

data=pd.read_csv('TP/dataset/classification/labeling_data.csv', encoding='utf-8')

sns.distplot(a=data[data.Churn==1]['MonthlyRevenue'],hist=True, kde=False, rug=False)
plt.show()

sns.distplot(a=data[data.Churn==1]['MonthlyMinutes'],hist=True, kde=False, rug=False)
plt.show()

sns.distplot(a=data[data.Churn==1]['TotalRecurringCharge'],hist=True, kde=False, rug=False)
plt.show()

sns.distplot(a=data[data.Churn==1]['DirectorAssistedCalls'],hist=True, kde=False, rug=False)
plt.show()

sns.distplot(a=data[data.Churn==1]['OverageMinutes'],hist=True, kde=False, rug=False)
plt.show()

sns.distplot(a=data[data.Churn==1]['RoamingCalls'],hist=True, kde=False, rug=False)
plt.show()

sns.distplot(a=data[data.Churn==1]['PercChangeMinutes'],hist=True, kde=False, rug=False)
plt.show()

sns.distplot(a=data[data.Churn==1]['PercChangeRevenues'],hist=True, kde=False, rug=False)
plt.show()

sns.distplot(a=data[data.Churn==1]['DroppedCalls'],hist=True, kde=False, rug=False)
plt.show()

sns.distplot(a=data[data.Churn==1]['OverageMinutes'],hist=True, kde=False, rug=False)
plt.show()

sns.distplot(a=data[data.Churn==1]['BlockedCalls'],hist=True, kde=False, rug=False)
plt.show()

sns.distplot(a=data[data.Churn==1]['UnansweredCalls'],hist=True, kde=False, rug=False)
plt.show()

sns.distplot(a=data[data.Churn==1]['CustomerCareCalls'],hist=True, kde=False, rug=False)
plt.show()

sns.distplot(a=data[data.Churn==1]['ThreewayCalls'],hist=True, kde=False, rug=False)
plt.show()

sns.distplot(a=data[data.Churn==1]['ReceivedCalls'],hist=True, kde=False, rug=False)
plt.show()

sns.distplot(a=data[data.Churn==1]['OutboundCalls'],hist=True, kde=False, rug=False)
plt.show()

sns.distplot(a=data[data.Churn==1]['InboundCalls'],hist=True, kde=False, rug=False)
plt.show()

sns.distplot(a=data[data.Churn==1]['PeakCallsInOut'],hist=True, kde=False, rug=False)
plt.show()

sns.distplot(a=data[data.Churn==1]['OffPeakCallsInOut'],hist=True, kde=False, rug=False)
plt.show()

sns.distplot(a=data[data.Churn==1]['DroppedBlockedCalls'],hist=True, kde=False, rug=False)
plt.show()

sns.distplot(a=data[data.Churn==1]['CallForwardingCalls'],hist=True, kde=False, rug=False)
plt.show()

sns.distplot(a=data[data.Churn==1]['CallWaitingCalls'],hist=True, kde=False, rug=False)
plt.show()

sns.distplot(a=data[data.Churn==1]['MonthsInService'],hist=True, kde=False, rug=False)
plt.show()

sns.distplot(a=data[data.Churn==1]['UniqueSubs'],hist=True, kde=False, rug=False)
plt.show()

sns.distplot(a=data[data.Churn==1]['ActiveSubs'],hist=True, kde=False, rug=False)
plt.show()

sns.distplot(a=data[data.Churn==1]['Handsets'],hist=True, kde=False, rug=False)
plt.show()

sns.distplot(a=data[data.Churn==1]['CurrentEquipmentDays'],hist=True, kde=False, rug=False)
plt.show()

sns.distplot(a=data[data.Churn==1]['Age'],hist=True, kde=False, rug=False)
plt.show()

sns.distplot(a=data[data.Churn==1]['ChildrenInHH'],hist=True, kde=False, rug=False)
plt.show()

sns.distplot(a=data[data.Churn==1]['HandsetRefurbished'],hist=True, kde=False, rug=False)
plt.show()

sns.distplot(a=data[data.Churn==1]['HandsetWebCapable'],hist=True, kde=False, rug=False)
plt.show()

sns.distplot(a=data[data.Churn==1]['TruckOwner'],hist=True, kde=False, rug=False)
plt.show()

sns.distplot(a=data[data.Churn==1]['RVOwner'],hist=True, kde=False, rug=False)
plt.show()

sns.distplot(a=data[data.Churn==1]['Homeownership'],hist=True, kde=False, rug=False)
plt.show()

sns.distplot(a=data[data.Churn==1]['BuysViaMailOrder'],hist=True, kde=False, rug=False)
plt.show()

sns.distplot(a=data[data.Churn==1]['RespondsToMailOffers'],hist=True, kde=False, rug=False)
plt.show()

sns.distplot(a=data[data.Churn==1]['OptOutMailings'],hist=True, kde=False, rug=False)
plt.show()

sns.distplot(a=data[data.Churn==1]['NonUSTravel'],hist=True, kde=False, rug=False)
plt.show()

sns.distplot(a=data[data.Churn==1]['OwnsComputer'],hist=True, kde=False, rug=False)
plt.show()

sns.distplot(a=data[data.Churn==1]['HasCreditCard'],hist=True, kde=False, rug=False)
plt.show()

sns.distplot(a=data[data.Churn==1]['RetentionCalls'],hist=True, kde=False, rug=False)
plt.show()

sns.distplot(a=data[data.Churn==1]['RetentionOffersAccepted'],hist=True, kde=False, rug=False)
plt.show()

sns.distplot(a=data[data.Churn==1]['NewCellphoneUser'],hist=True, kde=False, rug=False)
plt.show()

sns.distplot(a=data[data.Churn==1]['NotNewCellphoneUser'],hist=True, kde=False, rug=False)
plt.show()

sns.distplot(a=data[data.Churn==1]['ReferralsMadeBySubscriber'],hist=True, kde=False, rug=False)
plt.show()

sns.distplot(a=data[data.Churn==1]['IncomeGroup'],hist=True, kde=False, rug=False)
plt.show()

sns.distplot(a=data[data.Churn==1]['OwnsMotorcycle'],hist=True, kde=False, rug=False)
plt.show()

sns.distplot(a=data[data.Churn==1]['AdjustmentsToCreditRating'],hist=True, kde=False, rug=False)
plt.show()

sns.distplot(a=data[data.Churn==1]['HandsetPrice'],hist=True, kde=False, rug=False)
plt.show()

sns.distplot(a=data[data.Churn==1]['MadeCallToRetentionTeam'],hist=True, kde=False, rug=False)
plt.show()

sns.distplot(a=data[data.Churn==1]['CreditRating'],hist=True, kde=False, rug=False)
plt.show()

sns.distplot(a=data[data.Churn==1]['PrizmCode'],hist=True, kde=False, rug=False)
plt.show()

sns.distplot(a=data[data.Churn==1]['Occupation'],hist=True, kde=False, rug=False)
plt.show()

sns.distplot(a=data[data.Churn==1]['MaritalStatus'],hist=True, kde=False, rug=False)
plt.show()
