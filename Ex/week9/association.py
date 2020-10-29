import numpy as np
import matplotlib.pyplot as plt  
import pandas as pd
from apyori import apriori

movie_data = pd.read_csv('C:/Users/hays3/git/CyberSecurityHw/dataSet.csv',  header = None)
num_records = len(movie_data)  
print(num_records)

records = []
for i in range(0, num_records):  
    records.append([str(movie_data.values[i,j]) for j in range(0, 20)])

association_rules = apriori(records, min_support=0.0053,  min_confidence=0.80, min_lift=3, min_length=2)
# Convert the rules found by the apriori class into a list  to make it easier to view.
association_results = list(association_rules)

print(len(association_results))  
print(association_results)
