from typing import NewType
from numpy.core.numeric import indices
import pandas as pd 
import math 
import numpy as np
from sklearn.linear_model import LinearRegression as LR
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt 
from sklearn.neighbors import (NeighborhoodComponentsAnalysis,
KNeighborsClassifier)
from sklearn.pipeline import Pipeline

data = pd.read_csv('C:/Users/jdbai/Desktop/heart.csv')
#Separate Data 
x = data.drop('HeartDisease', axis =1)
y = data['HeartDisease']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2)

y_test.to_csv('C:/Users/jdbai/Desktop/y_test_svm.csv')
data2 = pd.read_csv('C:/Users/jdbai/Desktop/y_test_svm.csv', index_col=False)
new_test = data2['HeartDisease']

nca = NeighborhoodComponentsAnalysis(random_state=42)
knn = KNeighborsClassifier(n_neighbors=3)
nca_pipe = Pipeline([('nca', nca), ('knn', knn)])
nca_pipe.fit(x_train, y_train)

y_test.reset_index(drop=True, inplace=True)








plt.plot(y_test,'ro')
plt.plot(nca_pipe,'bx')
plt.show()

#MIN MAX Scaling 
classified = [[0, 0], [0, 0]]
unclassified = 0
classed = 0
temp = 0

for x in range (len(new_test)):
    temp = new_test.loc[new_test[x]]
    if nca_pipe[x] == 1:
        classed += 1
        if nca_pipe[x] == temp:
            classified[0][0] += 1 #True Positive 
        else:
            classified[0][1] += 1 #False Positive   
    elif nca_pipe[x] ==0:
        classed += 1
        if nca_pipe [x]== temp:
            classified[1][1] += 1 #True Negative  
        else:
            classified[1][0] += 1 #False Negative 
    else: 
        unclassified += 1

plt.plot(y_test, 'ro')
plt.plot(nca_pipe, 'bx')
plt.show(); 

HD_yes = 0
HD_no = 0
for x in range (len(new_test)):
    temp = new_test.loc[new_test[x]]
    print(x ," ", temp )
    if temp > 0.5:
        HD_yes += 1 
        print("HD Yes")
    elif temp < 0.5:
        HD_no += 1
        print("HD No")
    else: 
        print("ERROR 1")

fig = plt.figure(figsize=(10,5))
results = ['HD_Yes', 'HD_No','TP', 'FP', 'TN', 'FN','C','UC']
numResults = [HD_yes, HD_no, classified[0][0],classified[0][1],classified[1][0],classified[1][1],classed,unclassified]
plt.bar(results,numResults)
plt.show()

print("Test Data with HD: ", HD_yes)
print("Test Data without HD:", HD_no)
print("True Positive: ", classified[0][0])
print("False Positive: ", classified[0][1])
print("True Negative: ", classified[1][0])
print("False Negative: ", classified[1][1])
print("Unclassified: ", unclassified)

