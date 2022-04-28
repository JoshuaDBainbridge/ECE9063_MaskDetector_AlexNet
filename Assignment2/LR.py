from numpy.core.numeric import indices
import pandas as pd 
import math 
import numpy as np
from sklearn.linear_model import LinearRegression as LR
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt 

data = pd.read_csv('C:/Users/jdbai/Desktop/heart.csv')
#Separate Data 
x = data.drop('HeartDisease', axis =1)
y = data['HeartDisease']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2)

y_test.to_csv('C:/Users/jdbai/Desktop/y_test.csv')
data2 = pd.read_csv('C:/Users/jdbai/Desktop/y_test.csv', index_col=False)
new_test = data2['HeartDisease']

HD_yes = 0
HD_no = 0

for x in range (len(new_test)):
    temp = new_test.loc[new_test[x]]
    if temp > 0:
        HD_yes += 1 
    else:
        HD_no += 1

# creating an object of LinearRegression class
model = LR()
# fitting the training data
model.fit(x_train,y_train)


y_prediction = np.array(model.predict(x_test))
y_predictionNEW =y_prediction.reshape(len(y_prediction),1)
y_test.reset_index(drop=True, inplace=True)


plt.plot(y_test, 'ro')
plt.plot(y_predictionNEW, 'bx')
plt.show(); 

#MIN MAX Scaling 
minMax_Y = y_predictionNEW
minMaxRange = 0.3

classified = [[0, 0], [0, 0]]
unclassified = 0
classed = 0

for x in range (len(minMax_Y)):
    temp = new_test.loc[new_test[x]]
    if minMax_Y[x] >= 1- minMaxRange:
        minMax_Y[x] = 1
        classed += 1
        if minMax_Y[x] == temp:
            classified[0][0] += 1 #True Positive 
        else:
            classified[0][1] += 1 #False Positive   
    elif minMax_Y[x] <= minMaxRange:
        minMax_Y[x] = 0
        classed += 1
        if minMax_Y[x]== temp:
            classified[1][1] += 1 #True Negative  
        else:
            classified[1][0] += 1 #False Negative 
    else: 
        minMax_Y[x] = 0.5
        unclassified += 1

plt.plot(y_test, 'ro')
plt.plot(minMax_Y, 'bx')
plt.show(); 


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