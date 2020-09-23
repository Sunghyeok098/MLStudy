from sklearn.model_selection import train_test_split  
from sklearn import datasets
from sklearn.tree import DecisionTreeRegressor


x_training_set, x_test_set, y_training_set, y_test_set =  train_test_split(X,y,test_size=0.10,random_state=42,  shuffle=True)

model = DecisionTreeRegressor(max_depth=5, random_state=0)  
model.fit(x_training_set, y_training_set)
