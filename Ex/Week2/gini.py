import pandas as pd
import numpy as np
import seaborn as sns
from sklearn import tree
from sklearn.model_selection import train_test_split

data_url = 'D:\VSCODE\MLStudy\Ex\Week2\dataSet1.xlsx'
dataSet = pd.read_excel(data_url)

data = pd.DataFrame(dataSet)

features = data[['Novelist','Novel Genre']]
target = data['Hit']

print(data, '\n')


#calculate root node entropy 
def entropy(target_col):
    #take element and each elements counting
    elements, counts = np.unique(target_col, return_counts = True)
    print(elements, counts)
    #calculate root node entropy 
    #entropy = -np.sum([(counts[i]/np.sum(counts))*np.log2(counts[i]/np.sum(counts)) for i in range(len(elements))])
    entropy = 1 - np.sum([(counts[i]/np.sum(counts))*(counts[i]/np.sum(counts)) for i in range(len(elements))])
    print('abcd',entropy)
    return entropy

# calculate information gain
def InfoGain(data,split_attribute_name,target_name):
 
    #calculate root node entropy 
    total_entropy = entropy(data[target_name])
    print('Entropy(D) = ', round(total_entropy, 5))
    
    #calculate next level node entropy 
    vals,counts= np.unique(data[split_attribute_name],return_counts=True)
    print(vals)
    print(counts)
    Weighted_Entropy = np.sum([(counts[i]/np.sum(counts))*
                               entropy(data.where(data[split_attribute_name]==vals[i]).dropna()[target_name])
                               for i in range(len(vals))])
    print('H(', split_attribute_name, ') = ', round(Weighted_Entropy, 5))
 
    print("weight : ", Weighted_Entropy )
    # calculate information gain
    Information_Gain = total_entropy - Weighted_Entropy
    return Information_Gain

def ID3(data,originaldata,features,target_attribute_name,parent_node_class = None):
     
    # If the destination property has a single value: return the destination property
    if len(np.unique(data[target_attribute_name])) <= 1:
        return np.unique(data[target_attribute_name])[0]
 
    # When there is no data: Returns the target property with the maximum value from the source data
    elif len(data)==0:
        return np.unique(originaldata[target_attribute_name])\
               [np.argmax(np.unique(originaldata[target_attribute_name], return_counts=True)[1])]
 
    # When no technical properties exist: Return destination properties of parent node
    elif len(features) ==0:
        return parent_node_class
 
    # tree growth
    else:
        # Define target properties for the parent node
        parent_node_class = np.unique(data[target_attribute_name])\
                            [np.argmax(np.unique(data[target_attribute_name], return_counts=True)[1])]
        
        # Select properties to split data
        item_values = [InfoGain(data,feature,target_attribute_name) for feature in features]
    
        best_feature_index = np.argmax(item_values)
        best_feature = features[best_feature_index]
        print('Best InfoGain(', best_feature,  ') = ', max(item_values), '\n')

     
        # tree structure generation
        tree = {best_feature:{}}
        
        # Exclude technical attributes that show Best information
        features = [i for i in features if i != best_feature]
        
        # branch growth
        for value in np.unique(data[best_feature]):
            # rows with missing values, removing columns
            sub_data = data.where(data[best_feature] == value).dropna()
            
            subtree = ID3(sub_data,data,features,target_attribute_name,parent_node_class)
            tree[best_feature][value] = subtree
            
        return(tree)


tree = ID3(data, data, ['Novelist','Novel Genre'], 'Hit')
from pprint import pprint
pprint(tree)