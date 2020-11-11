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