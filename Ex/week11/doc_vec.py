from sklearn.feature_extraction.text import CountVectorizer  
from sklearn.feature_extraction.text import TfidfTransformer  
import numpy as np
import pandas as pd
import re 

sentences = list() 

with open("Ex/dataSet/week11_dataSet/", encoding='UTF-8') as file:

    for line in file:
        for l in re.split(r"\.\s|\?\s|\!\s|\n",line):  
            if l:
                sentences.append(l)

print(sentences)


from numpy import matrix
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import pandas as pd
import re

sentences = list()
with open("resources/dataset.txt") as file:
    for line in file:
        for l in re.split(r"\.\s|\?\s|\!\s|\n",line):
            if l:
                sentences.append(l)
print(sentences)

# create a vectorizer model
vectorizer = TfidfVectorizer()
# fit the model to train data to create a normalized TF-IDF



from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
import numpy as np
f = open("resources/d1.txt", 'r')
data1 = f.read()
print(data1)

f = open("resources/d2.txt", 'r')
data2 = f.read()
print(data2)

f = open("resources/d3.txt", 'r')
data3 = f.read()
print(data3)

f = open("resources/d4.txt", 'r')
data4 = f.read()
print(data4)

f = open("resources/d5.txt", 'r')
data5 = f.read()
print(data5)

f.close()

tfidf_vectorizer = TfidfVectorizer()

query = ["corona vaccine fake news"]
data1 = [data1]
data2 = [data2]
data3 = [data3]
data4 = [data4]
data5 = [data5]

text = query + data1 + data2 + data3 + data4 + data5

print(text)

tfidf_vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf_vectorizer.fit_transform(text)
#Print Vocabulary
print(tfidf_vectorizer.vocabulary_)
#shape print word frequently
print(tfidf_matrix.shape)
#TF-IDF TABLE
print(tfidf_matrix.toarray())

from sklearn.metrics.pairwise import linear_kernel
#cosine similarity
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
print(cosine_sim)
data = pd.DataFrame(cosine_sim,columns=["Q","D1","D2","D3","D4","D5"]);
data2 = data.iloc[[0],:]
data2 = data2.iloc[:,1:]
#rank order
print(data2.rank(method='max', ascending=False, axis=1))