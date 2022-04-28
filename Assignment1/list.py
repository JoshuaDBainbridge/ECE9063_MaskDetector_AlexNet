import pandas as pd 
import numpy  as np 
df = pd.read_excel("C:/Users/jdbai/Desktop/heartData.xlsx", sheet_name=0) 
mylist_CP = df['ChestPainType'].tolist()
mylist_RE = df['RestingECG'].tolist()
mylist_SS = df['ST_Slope'].tolist()
#this play list 
#print(mylist)

#find unique values from CP  
unique_CP = list(dict.fromkeys(mylist_CP))
print(unique_CP)
occur_CP = [0,0,0,0]
for x in range (len(mylist_CP)): 
    if mylist_CP[x] == unique_CP[0]:
        occur_CP[0] +=1; 
    elif mylist_CP[x] == unique_CP[1]:
        occur_CP[1] +=1; 
    elif mylist_CP[x] == unique_CP[2]:
        occur_CP[2] +=1; 
    elif mylist_CP[x] == unique_CP[3]:
        occur_CP[3] +=1;         
#print(occur_CP)

#find unique values from RE 
unique_RE = list(dict.fromkeys(mylist_RE))
print(unique_RE)

#find unique values from SS
unique_SS = list(dict.fromkeys(mylist_SS))
print(unique_SS)

