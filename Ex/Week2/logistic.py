
from sklearn.datasets import fetch_openml
mnist = fetch_openml('mnist_784')

from sklearn.model_selection import train_test_split


train_img, test_img, train_lbl, test_lbl =  train_test_split(mnist.data, mnist.target, test_size=1/7.0, random_state=0)
from sklearn.linear_model import LogisticRegression
# all parameters not specified are set to their  defaults
# default solver is too slow, so we change it to  
# # Broyden-Fletcher-Goldfarb-Shanno algorithm
logisticRegr = LogisticRegression(solver = 'lbfgs')
logisticRegr.fit(train_img, train_lbl)


# Returns a NumPy Array
# Predict for One Observation (image)  
logisticRegr.predict(test_img[0].reshape(1,-1))
#Predict multiple observations (images)  
logisticRegr.predict(test_img[0:10])
predictions = logisticRegr.predict(test_img)
# Use the score method to get the accuracy of  model
score = logisticRegr.score(test_img, test_lbl)  
print(score)
