# Random Forest
# Created By Mitakus Team 7 (Moez-Mouath-Salma)
# Importing the libraries
import numpy as np
import pandas as pd
# importing some libraries for visulizations
import matplotlib.pyplot as plt
import seaborn as sns


#**********************************
# Importing the dataset
dataset15Aggregated = pd.read_csv('../Data2015Aggregated - CONFIDENTIAL.csv',sep=';')
dataset16Aggregated = pd.read_csv('../Data2016Aggregated - CONFIDENTIAL.csv',sep=';')

#***********************************
#visualizing the ID
#this function will return a dictionary where its Keys will be items' ID and its values will be the total number of guest along the year
def getIdItem(data,column):
    dic={}
    for i in range(data.shape[0]):
        for j in range(len(data[column].str.split('|')[i])):
          key = str(data[column].str.split('|')[i][j].split('>')[0])  
          if  key != 'Nan' and not key in dic.keys() :
            dic[key]=0
          elif key in dic.keys() :
            dic[key]+=int(data[column].str.split('|')[i][j].split('>')[1])
    return dic

##visualize the top 10 ID for each kind of dish in 2015 and 2016
def visualize_Dish():    
    for col in ['E_GreenDish','F_YellowDish','G_RedDish']:
      i=0
      for data in [dataset15Aggregated,dataset16Aggregated]:
        dic=getIdItem(data,col)
        plt.figure(figsize=(30,20))
        plt.title(col + 'in ' + str(2015+i))
        plt.xlabel('Id Items')
        plt.ylabel('number of Guests')
        xs = sorted(dic.values(), reverse=True)[:10]
        ys = [list(dic.keys())[list(dic.values()).index(x)] for x in  xs]
        plt.bar(ys, xs)
        i+=1

#visualize_Dish()


def transform(datasetAggregated):
    global plt,pd,np,sns
    #******************************************
    #change the format of the column 'A_Day' to datetime
    datasetAggregated.A_Day=pd.to_datetime(datasetAggregated.A_Day)
    year = str(datasetAggregated.A_Day[1].year)
    
    
    #******************************************
    #drop all the rows where the revenue equals to zero
    datasetAggregated = datasetAggregated[datasetAggregated["B_Revenue"] != '0,0']
    datasetAggregated = datasetAggregated.reset_index(drop=True)
    
    
    #******************************************
    #Create two columns for the days' and months' value
    datasetAggregated['Days'] = [x.strftime("%w") for x in datasetAggregated.A_Day]
    datasetAggregated['Days'] = datasetAggregated['Days'].astype(int)
    
    datasetAggregated['Month'] = [x.month for x in datasetAggregated.A_Day]
        
        
    #******************************************
    #Encoding the data using the dummy variabl methode for the days and months
    datasetAggregated = pd.get_dummies(datasetAggregated, columns=['Days'], drop_first=True, dtype=int)
#    datasetAggregated = pd.get_dummies(datasetAggregated, columns=['Month'], drop_first=False, dtype=int)
    
    
    #******************************************
    #replace the Nan value in E_GreenDish, F_YellowDish and G_RedDish by '0'
    for x in ['E_GreenDish','G_RedDish','F_YellowDish']:
        datasetAggregated.loc[datasetAggregated[x] == 'Nan', x] = '0'
    
    
    #******************************************
    #deal with the dishes
    Col=['E_GreenDish','G_RedDish','F_YellowDish']
    Dish=['Green','Red','Yellow']
    for x in range(3):
        datasetAggregated['Numberof'+Dish[x]+'DishID']=datasetAggregated[Col[x]].str.split('|').str.len()
        for i in range(max(datasetAggregated[Col[x]].str.split('|').str.len())):
            datasetAggregated[Dish[x]+str(i+1)]=np.zeros((datasetAggregated.shape[0],1))
            datasetAggregated['Guest_'+Dish[x]+str(i+1)]=np.zeros((datasetAggregated.shape[0],1))
            

    for x in range(3):
        for i in range(max(datasetAggregated[Col[x]].str.split('|').str.len())):   
            datasetAggregated[Dish[x]+str(i+1)]=datasetAggregated[Col[x]].str.split('|').str[i].str.split('>').str[0]
            datasetAggregated['Guest_'+Dish[x]+str(i+1)]=datasetAggregated[Col[x]].str.split('|').str[i].str.split('>').str[1]
            datasetAggregated.fillna(0, inplace=True)
            datasetAggregated[Dish[x]+str(i+1)] = datasetAggregated[Dish[x]+str(i+1)].astype(int)
            datasetAggregated['Guest_'+Dish[x]+str(i+1)] = datasetAggregated['Guest_'+Dish[x]+str(i+1)].astype(int)
            
    datasetAggregated.fillna(0, inplace=True)
    datasetAggregated=datasetAggregated.reset_index(drop=True)
#    G=datasetAggregated.Guest_Green1 + datasetAggregated.Guest_Green2 + datasetAggregated.Guest_Green3 + datasetAggregated.Guest_Green4
#    R=datasetAggregated.Guest_Red1 + datasetAggregated.Guest_Red2 + datasetAggregated.Guest_Red3 + datasetAggregated.Guest_Red4
#    Y=datasetAggregated.Guest_Yellow1 + datasetAggregated.Guest_Yellow2 + datasetAggregated.Guest_Yellow3
#    datasetAggregated['Others']=datasetAggregated.C_GuestsBreakfast + datasetAggregated.D_GuestsLunch - G - R - Y
    
    datasetAggregated=datasetAggregated.drop(['A_Day','B_Revenue','F_YellowDish','G_RedDish','E_GreenDish'],axis=1)
    return datasetAggregated

dataset15Aggregated=transform(dataset15Aggregated)
dataset16Aggregated=transform(dataset16Aggregated)

Data= dataset16Aggregated.append(dataset15Aggregated, ignore_index=True)
Data.fillna(0, inplace=True)

Data.columns
Green1 = Data[['NumberofGreenDishID','Days_2', 'Days_3', 'Days_4','Days_5','Month','Green1','Guest_Green1']]
Green1 = Green1[Green1["Green1"] != 0]

X=Green1.iloc[:, :-1].values
Y=Green1.iloc[:, -1].values



# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_Train, X_Test, Y_Train, Y_Test = train_test_split(X, Y, test_size = 0.2, random_state=0)

from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 1000)
regressor.fit(X_Train, Y_Train)

# Predicting a new result
y_pred = regressor.predict(X_Test)

from sklearn.metrics import accuracy_score
y_pred = y_pred.astype(int)
acc=accuracy_score(y_pred,Y_Test)
