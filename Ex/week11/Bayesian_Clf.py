
flu = ['Y', 'N', 'Y', 'Y', 'Y', 'N', 'Y', 'Y', 'Y', 'N', 'Y', 'Y','Y', 'N', 'Y', 'Y', 'Y', 'N', 'Y', 'Y', 'Y', 'N', 'Y', 'Y','Y', 'N', 'Y', 'Y', 'Y', 'N']
fever = ['L', 'M', 'H', 'M','L', 'M', 'H', 'M','L', 'M', 'H', 'M','L', 'M', 'H', 'M','L', 'M', 'H', 'M','L', 'M', 'H', 'M','L', 'M', 'H', 'M','L', 'M']
sinus = ['Y', 'N', 'Y', 'Y', 'Y', 'N', 'Y', 'Y','Y', 'N', 'Y', 'Y','Y', 'N', 'Y', 'Y','Y', 'N', 'Y', 'Y','Y', 'N', 'Y', 'Y','Y', 'N', 'Y', 'Y','Y', 'N']
ache = ['Y', 'N', 'N', 'N','Y', 'N', 'N', 'N','Y', 'N', 'N', 'N','Y', 'N', 'N', 'N','Y', 'N', 'N', 'N','Y', 'N', 'N', 'N','Y', 'N', 'N', 'N','Y', 'N']
swell = ['Y', 'N', 'Y', 'N','Y', 'N', 'Y', 'N','Y', 'N', 'Y', 'N','Y', 'N', 'Y', 'N','Y', 'N', 'Y', 'N','Y', 'N', 'Y', 'N','Y', 'N', 'Y', 'N','Y', 'N']
headache = ['N', 'N', 'Y', 'Y', 'N', 'N','Y', 'Y', 'N', 'N','Y', 'Y', 'N', 'N','Y', 'Y', 'N', 'N','Y', 'Y', 'N', 'N', 'Y', 'Y', 'N', 'N','Y', 'Y', 'N', 'N'] 

# Import LabelEncoder
from sklearn import preprocessing

#create labelEncoder
le = preprocessing.LabelEncoder()

# Convert string labels into numbers.  
label = le.fit_transform(flu)
fever_encoded = le.fit_transform(fever)
sinus_encoded = le.fit_transform(sinus)
ache_encoded = le.fit_transform(ache)
swell_encoded = le.fit_transform(swell)
headache_encoded = le.fit_transform(headache)

features = list(zip(fever_encoded, sinus_encoded, ache_encoded, swell_encoded, headache_encoded))
#print(features)

#Import Gaussian Naive Bayes model
from sklearn.naive_bayes import GaussianNB  #Create a Gaussian Classifier
model = GaussianNB()

# Train the model using the training sets  
model.fit(features, label)
#Predict Output
predicted= model.predict([[0,1,1,1,1]]) # fever:H, sinus:Y, ache:Y, swell:Y, headache:Y   

print('Predicted Value : ', predicted) 

if(predicted[0] == 1):
    print('Predicted flu')
else:
    print('Predicted Not flu')


