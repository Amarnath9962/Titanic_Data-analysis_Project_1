import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Loading the data in to the python

Titanic_data = pd.read_csv("Titanic.csv")
print("The Total number of Passengers travelled in Titanic : ",len(Titanic_data))
print(Titanic_data)


# Analysing the data into the python
sns.countplot(x= "Survived",data= Titanic_data)
# plt.show()


sns.countplot(x= "Survived",hue="Sex",data= Titanic_data)
# plt.show()


sns.countplot(x = "Survived",hue = "Pclass",data=Titanic_data)
##plt.show()

sns.countplot(x= "Sex",hue= "Survived",data= Titanic_data)
##plt.show()


Titanic_data["Age"].plot.hist()
##plt.show()


# Data wranging

Null = Titanic_data.isnull()
Null1= Titanic_data.isnull().sum()
print(Null1)
#print(Null)

sns.heatmap(Titanic_data.isnull(),yticklabels=False,cmap = "viridis")
######       HEAT MAP IS USED FOR IDENTIFYING THE NO OF NULL VALUES
##plt.show()



##sns.boxlot(x= "Pclass",hue= "Sex",data= Titanic_data)
##plt.show()
print(Titanic_data.columns)
pd.set_option("max_rows", 10, "max_columns", 12)
##print(Titanic_data.head())

##Titanic_data.drop("Cabin",axis= 1,inplace= True)
##print(Titanic_data)

# print(Titanic_data["Cabin"])
# sns.heatmap(Titanic_data.isnull(),yticklabels= False,cbar= False)
# plt.show()

##Titanic_data.dropna(inplace= True)
##print(Titanic_data.isnull().sum())
###print(Titanic_data.info())
# ##
#
#
# ##  Data wrangling
#
sex = pd.get_dummies(Titanic_data["Sex"], drop_first=True)
# # print(sex)
#
Hello  = pd.get_dummies(Titanic_data["Embarked"],drop_first=True)
print(Hello)
#
pclass = pd.get_dummies(Titanic_data["Pclass"], drop_first=True)
# # print(pclass)
#
# ##name =pd.get_dummies(Titanic_data["Name"])
# ##print(name)
#
titanic_data = pd.concat([Titanic_data, sex, pclass,Hello], axis=1)
print(titanic_data)
# print(titanic_data.columns)
titanic_data.drop(["Sex","Embarked", "Pclass", "Name", "Ticket", "Fare", "Cabin", "PassengerId", "Age"], axis=1,inplace = True)
print(titanic_data)
#
# # TO FIND THE TEST AND TRAIN AND ACCURACY
#
# X = titanic_data.drop("Survived", axis=1)
# Y = titanic_data["Survived"]
# # import sklearn
#
#
# from sklearn.model_selection import train_test_split
# from sklearn.model_selection import train_test_splitpr
#
# X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
# print(X_train.shape)
#
# # print(X_train)#, X_test , Y_train , Y_test)


# from sklearn.linear_model import LogisticRegression
#
# logmodel = LogisticRegression()
# logmodel.fit(X_train, Y_train)
#
# predictions = logmodel.predict(X_test)
# print(predictions)


# from sklearn.metrics import classification_report
# print(classification_report(Y_test,predictions))


