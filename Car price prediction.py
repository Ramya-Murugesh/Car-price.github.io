#!/usr/bin/env python
# coding: utf-8

# In[93]:


#import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
import os
from sklearn.preprocessing import LabelEncoder,StandardScaler,MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression,Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score,mean_squared_error,mean_absolute_error
import warnings
warnings.filterwarnings('ignore')


# In[94]:


#Load dataset
dataset = pd.read_csv('F:\Data science\My Project\car data.csv')
dataset


# In[95]:


# Find basic info
dataset.info()


# In[96]:


#Describe data
dataset.describe()


# In[97]:


#Check for duplicates
dataset.duplicated().sum()


# In[98]:


# Identify the duplicates
duplicates_mask = dataset.duplicated(keep = False)
duplicate_rows = dataset[duplicates_mask]
print(duplicate_rows)


# In[99]:


#Drop the duplicates
dataset = dataset.drop_duplicates(keep = 'first')


# In[100]:


#Print dataset
dataset


# In[101]:


dataset.count()


# In[102]:


# check for null values
dataset.isna().sum()


# In[103]:


# statistical analysis
dataset.describe()


# In[104]:


# check for outliers
sns.boxplot(dataset)
plt.show()


# #Kms_Driven shows outliers. Use MinMaxScaler for feature scaling

# #Let's find some unique value counts for object datatypes

# In[105]:


dataset['Car_Name'].unique()


# In[106]:


dataset['Car_Name'].value_counts()


# In[107]:


dataset['Fuel_Type'].unique()


# In[108]:


dataset['Fuel_Type'].value_counts()


# In[109]:


dataset['Seller_Type'].unique()


# In[110]:


dataset['Seller_Type'].value_counts()


# In[111]:


dataset['Transmission'].unique()


# In[112]:


dataset['Transmission'].value_counts()


# In[113]:


#Convert objects into numerical values using Label encoder
le = LabelEncoder()
for columns in dataset.columns:
    if dataset[columns].dtypes == object:
        dataset[columns] = le.fit_transform(dataset[columns])
dataset


# In[114]:


#check datatypes
dataset.dtypes


# In[115]:


#Let's find the correlation between the variables
dataset.corr()


# In[116]:


#heat map for visual correlation
plt.figure(figsize = (15,8))
sns.heatmap(dataset.corr(),annot =True)
plt.show()


# In[117]:


plt.figure(figsize = (15,8))
sns.boxplot(dataset)
plt.show()


# In[118]:


# Find relation between car name and selling price
sns.barplot(x = 'Car_Name', y = 'Selling_Price', data = dataset)
plt.show()


# #Car_Name doesn't have major impact on selling price except a few, so, let's drop the Car_Name

# In[119]:


#Drop Car_Name
dataset = dataset.drop(['Car_Name'],axis = 1)
dataset


# In[120]:


#convert the year to age of the car
dataset['Age'] = 2023 - dataset['Year']
dataset['Age']


# In[121]:


#print dataset
dataset


# In[122]:


#Drop year column
dataset = dataset.drop(['Year'],axis = 1)
dataset


# In[123]:


# visula representation of impact of age of cars on selling price
plt.figure(figsize = (15,8))
sns.barplot(x = 'Age', y = 'Selling_Price', data = dataset)
plt.xlabel('Age of cars')
plt.show()


# #Mostly when the age is less, the selling price is high

# In[124]:


#impact of kms driven on selling price
plt.figure(figsize = (15,8))
sns.barplot(x = 'Kms_Driven', y = 'Selling_Price', data = dataset)
plt.show()


# #Data visualization

# In[125]:


dataset.hist(color = '#7724DC',figsize = (10,8))
plt.show()


# In[126]:


plt.figure(figsize = (15,8))
sns.barplot(x = 'Fuel_Type', y = 'Selling_Price', data = dataset)
plt.show()


# In[127]:


plt.figure(figsize = (15,8))
sns.barplot(x = 'Seller_Type', y = 'Selling_Price', data = dataset)
plt.show()


# In[128]:


plt.figure(figsize = (15,8))
sns.barplot(x = 'Transmission', y = 'Selling_Price', data = dataset)
plt.show()


# In[129]:


plt.figure(figsize = (15,8))
sns.barplot(x = 'Owner', y = 'Selling_Price', data = dataset)
plt.show()


# In[130]:


dataset


# In[131]:


#Target variable(Dependent)
y = dataset['Selling_Price']
y


# In[132]:


#Independent variables
x = dataset.iloc[:,1:]
x


# In[133]:


#Feature scaling
mms = MinMaxScaler()
mms = mms.fit_transform(x)
mms


# In[134]:


print(pd.DataFrame(mms))


# In[135]:


#Split the data
x_train,x_test,y_train,y_test = train_test_split(mms,y,test_size = 0.2,random_state = 42)


# In[136]:


x_train.shape


# In[137]:


x_test.shape


# In[138]:


y_train.shape


# In[139]:


y_test.shape


# #Let's try various regression algorithms to find which model predicts better accuracy

# In[140]:


lr = LinearRegression()
lr.fit(x_train,y_train)
lr_pred = lr.predict(x_test)
lr_pred


# In[141]:


r2_score(y_test,lr_pred)


# In[142]:


mean_squared_error(y_test,lr_pred)


# In[143]:


mean_absolute_error(y_test,lr_pred)


# In[144]:


rr = Ridge()
rr.fit(x_train,y_train)
rr_pred = rr.predict(x_test)
rr_pred


# In[145]:


r2_score(y_test,rr_pred)


# In[146]:


mean_squared_error(y_test,rr_pred)


# In[147]:


mean_absolute_error(y_test,rr_pred)


# In[148]:


dt = DecisionTreeRegressor()
dt.fit(x_train,y_train)
dt_pred = dt.predict(x_test)
dt_pred


# In[149]:


r2_score(y_test,dt_pred)


# In[150]:


rf = RandomForestRegressor(n_estimators = 100)
rf.fit(x_train,y_train)
rf_pred = rf.predict(x_test)
rf_pred


# In[151]:


r2_score(y_test,rf_pred)


# In[152]:


x_train,x_test,y_train,y_test = train_test_split(mms,y,test_size = 0.2,random_state = 42)


# In[153]:


rf = RandomForestRegressor(n_estimators = 200)
rf.fit(x_train,y_train)
rf_pred = rf.predict(x_test)
rf_pred


# In[154]:


r2_score(y_test,rf_pred)


# In[155]:


x_train,x_test,y_train,y_test = train_test_split(mms,y,test_size = 0.2,random_state = 42)


# In[156]:


dt = DecisionTreeRegressor()
dt.fit(x_train,y_train)
dt_pred = dt.predict(x_test)
dt_pred


# In[157]:


r2_score(y_test,dt_pred)


# In[158]:


x_train,x_test,y_train,y_test = train_test_split(mms,y,test_size = 0.2,random_state = 42)


# In[159]:


rr = Ridge()
rr.fit(x_train,y_train)
rr_pred = rr.predict(x_test)
rr_pred


# In[160]:


r2_score(y_test,rr_pred)


# In[161]:


x_train,x_test,y_train,y_test = train_test_split(mms,y,test_size = 0.2,random_state = 42)


# In[162]:


xgb_reg = xgb.XGBRegressor(n_estimators = 2000,max_depth = 3, learning_rate = 0.1,random_state = 42)
xgb_reg.fit(x_train,y_train)
xgb_pred = xgb_reg.predict(x_test)
xgb_pred


# In[163]:


r2_score(y_test,xgb_pred)


# In[164]:


x_train,x_test,y_train,y_test = train_test_split(mms,y,test_size = 0.2,random_state = 42)


# In[165]:


knn = KNeighborsRegressor()
knn.fit(x_train,y_train)
knn_pred = knn.predict(x_test)
knn_pred


# In[166]:


r2_score(y_test,knn_pred)


# In[167]:


rmse_val = []
for k in range(20):
    k = k+1
    knn = KNeighborsRegressor(n_neighbors = k)
    knn.fit(x_train,y_train)
    knn_pred = knn.predict(x_test)
    acc = r2_score(y_test,knn_pred)
    rmse_val.append(acc)
    print('RMSE value for k = ', k, 'is: ', acc)
    plt.plot(range(1,21)),rmse_val


# In[175]:


knn = KNeighborsRegressor(n_neighbors = 1)
knn.fit(x_train,y_train)
knn_pred = knn.predict(x_test)
knn_pred


# In[169]:


r2_score(y_test,knn_pred)


# #KNeighborsRegressor gives better accuracy than other models
