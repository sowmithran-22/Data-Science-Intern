#!/usr/bin/env python
# coding: utf-8

# In[4]:


#DATA COLLECTION
import pandas as pd
import numpy as np 
#read the given dataset
df=pd.read_csv("50_Startups.csv")


# In[8]:


#DATA MANIPULATING  
#The DataFrame object by using head and tail
#head
df.head()


# In[9]:


#tail
df.tail()


# In[10]:


#isnull to check the existing of null values
df.isnull()


# In[11]:


#define the shape of the dataset
df.shape #rows,column


# In[12]:


# Remove missing values
df.dropna()


# In[13]:


#importing library for analysing data
import statsmodels.api as sm


# In[14]:


#describing the dataset 
df.describe()


# In[15]:


#DATA VISUALIZATION
#import libraries for data visualization
import matplotlib.pyplot as plt


# In[16]:


#Simple line plot
df.plot()


# In[17]:


#Histogram
df.plot(kind="hist")


# In[18]:


#Area plot
df.plot.area()


# In[19]:


#plotting using matplotlib 
#1.Scatter plot
#i)splitting variables

RD_spend = df['R&D Spend']
Admin_cost = df['Administration']
Marketing_spend = df['Marketing Spend']
Profit = df['Profit']


# In[20]:


# ii)Scatter plot using plt
plt.figure(figsize=(10, 6))
plt.scatter(RD_spend, Profit, label='R&D Spend')
plt.scatter(Admin_cost, Profit, label='Administration Cost')
plt.scatter(Marketing_spend, Profit, label='Marketing Spend')

plt.xlabel('Amount')
plt.ylabel('Profit')
plt.title('Scatter Plot of R&D Spend, Administration Cost, Marketing Spend, and Profit')
plt.legend()
plt.show()


# In[21]:


# iii)stem plot using matplotlib
plt.figure(figsize=(10, 6))
plt.stem(RD_spend, Profit, label='R&D Spend')
plt.stem(Admin_cost, Profit, label='Administration Cost')
plt.stem(Marketing_spend, Profit, label='Marketing Spend')

plt.xlabel('Amount')
plt.ylabel('Profit')
plt.title('Stem plot of R&D Spend, Administration Cost, Marketing Spend, and Profit')
plt.legend()
plt.show()


# In[22]:


plt.figure(figsize=(10, 6))
plt.polar(RD_spend, Profit, label='R&D Spend')
plt.polar(Admin_cost, Profit, label='Administration Cost')
plt.polar(Marketing_spend, Profit, label='Marketing Spend')

plt.xlabel('Amount')
plt.ylabel('Profit')
plt.title('Polar of R&D Spend, Administration Cost, Marketing Spend, and Profit')
plt.legend()
plt.show()


# In[23]:


#Visualization using Seaborn
#import libraries for data visualization
import seaborn as sns


# In[24]:


#distplot
sns.distplot(df,kde="True",color="m")


# In[25]:


#violin plot
sns.violinplot(data=df,split=True)


# In[26]:


#countplot
sns.countplot(data = df) 


# In[27]:


#Assigning variable to the values
X1=df["R&D Spend"]
X2=df["Administration"]
X3=df["Marketing Spend"]
Y=df["Profit"]


# In[28]:


#1. R&D Spend
#Percentage value for the R&D spend
df["R&D Spend"].quantile([0.1,0.25,0.50,0.75,1.0])


# In[29]:


#Relationship between X1(R&D Spend)
import matplotlib.pyplot as plt
plt.scatter(X1,Y,label='R&D Spend')
plt.xlabel('R&D Spend')
plt.ylabel('Profit')
plt.title('Scatter plot of R&D Spend and Profit')
plt.legend()
plt.show()


# In[30]:


#Condition for R&D spend and Profit
conditions = [
    (df['R&D Spend'] == 0),
    (df['R&D Spend'] > 0) & (df['R&D Spend'] <= 39936.370000),
    (df['R&D Spend'] > 39936.370000) & (df['R&D Spend'] <= 73051.080000),
    (df['R&D Spend'] > 73051.080000) & (df['R&D Spend'] <= 101602.800000),
    (df['R&D Spend'] > 101602.800000)
    ]


# In[31]:


#Assigning value to the given condition
values = ['0','0-25','25-50','50-75','75-100']


# In[32]:


# Assigning R&D Bin column for the reference
df['R&D Bin'] = np.select(conditions, values)
df.head()


# In[33]:


#displaying the profit value from R&D spend
a1 = pd.DataFrame(df.groupby(['R&D Bin'])['Profit'].mean())
a1


# In[34]:


#2.Adminstration
#Percentage value for the administartion 
df["Administration"].quantile([0.1,0.25,0.50,0.75,1.0])


# In[35]:


#Relationship between X2(Administration)
import matplotlib.pyplot as plt

plt.scatter(X2,Y,label='Administration',color="red")
plt.xlabel('Administration')
plt.ylabel('Profit')
plt.title('Scatter plot of Administration and Profit')
plt.legend()
plt.show()


# In[36]:


#Condition for Administration and Profit
conditions = [
    (df['Administration'] == 0),
    (df['Administration'] > 0) & (df['Administration'] <= 103730.875),
    (df['Administration'] > 103730.875) & (df['Administration'] <= 122699.795),
    (df['Administration'] > 122699.795) & (df['Administration'] <= 144842.180),
    (df['Administration'] > 144842.180)
    ]


# In[37]:


#Assigning value to the given condition
values = ['0','0-25','25-50','50-75','75-100']


# In[38]:


# Assigning Administration Bin column for the reference
df['Administration Bin'] = np.select(conditions, values)
df.head()


# In[39]:


#displaying the profit value from Administration Bin
a2 = pd.DataFrame(df.groupby(['Administration Bin'])['Profit'].mean())
a2


# In[40]:


#3. Marketing Spend
#Percentage value for the Marketing Spend
df["Marketing Spend"].quantile([0.1,0.25,0.50,0.75,1.0])


# In[41]:


#Relationship between X3(Marketing Spend)
import matplotlib.pyplot as plt
plt.scatter(X3,Y,label='Marketing Spend',color="purple")
plt.xlabel('Marketing Spend')
plt.ylabel('Profit')
plt.title('Scatter plot of Marketing Spend and Profit')
plt.legend()
plt.show()


# In[42]:


conditions = [
    (df['Marketing Spend'] == 0),
    (df['Marketing Spend'] > 0) & (df['Marketing Spend'] <= 129300.132500),
    (df['Marketing Spend'] > 129300.132500) & (df['Marketing Spend'] <= 212716.240000),
    (df['Marketing Spend'] > 212716.240000) & (df['Marketing Spend'] <= 299469.085000),
    (df['Marketing Spend'] > 299469.085000)
    ]


# In[43]:


#Assigning value to the given condition
values = ['0','0-25','25-50','50-75','75-100']


# In[44]:


# Assigning Marketing Spend Bin column for the reference
df['Marketing Bin'] = np.select(conditions, values)
df.head()


# In[60]:


#displaying the profit value from Marketing Bin
a3 = pd.DataFrame(df.groupby(['Marketing Bin'])["Profit"].mean())
a3


# In[151]:



import statsmodels.api as sm
import numpy as np

y = df["Profit"]
x_vars = ["R&D Spend","Administration","Marketing Spend"]
x = df[x_vars]
x = sm.add_constant(x)
model = sm.OLS(y, x)
results = model.fit()
print(results.summary())


# In[152]:


#TRAINING MODEL USING REGRESSION ALGORITHMS
#1.LINEAR REGRESSION
#import Linear regression library and train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


# In[153]:


#Assigning values for independent and dependent varaibles
x_vars = ["R&D Spend","Administration","Marketing Spend"]
x = df[x_vars]
Y=Y


# In[154]:


#print the Shape of X and Y
print("Shape of X:",x.shape)
print("Shape of Y:",Y.shape)


# In[155]:


#Split the training and testing data
x_train, x_test, Y_train, Y_test = train_test_split(x, Y, test_size=0.25, random_state=99)


# In[156]:


#print X_train and X_test
print("X_train:",x_train)
print("X_test:",x_test)


# In[157]:


#print Y_train and Y_test
print("Y_train:",Y_train)
print("Y_test:",Y_test)


# In[158]:


#print the shape of X_train and X_test
print("Shape of X_train",x_train.shape)
print("Shape of X_test",x_test.shape)


# In[159]:


#print the shape of Y_train and Y_test
print("Shape of Y_train",Y_train.shape)
print("Shape of Y_test",Y_test.shape)


# In[161]:


#train the model
regressor = LinearRegression()
regressor.fit(x_train, Y_train)


# In[162]:


# Make predictions on the test data
Y_predict = regressor.predict(x_test)


# In[163]:


#CALCULATING REGRESSION METRICS
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
mae_linear=mean_absolute_error(Y_predict,Y_test)
mse_linear=mean_squared_error(Y_predict,Y_test)
r2_linear=r2_score(Y_test, Y_predict)


# In[164]:


print("Mean_absolute_error of Linear model:",mean_absolute_error(Y_predict,Y_test))
print("Mean_squared_error of Linear model:",mean_squared_error(Y_predict,Y_test))
print("R2 value of Linear model is:", r2_score(Y_test, Y_predict))


# In[165]:


#2.RANDOM FOREST REGRESSOR
#Import necessary library
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split


# In[166]:


#Assigning values for independent and dependent varaibles
x_vars = ["R&D Spend","Administration","Marketing Spend"]
x = df[x_vars]
Y=Y


# In[78]:


#train_test_split
x_train,x_test,Y_train,Y_test=train_test_split(x,Y,test_size=0.25,random_state=99)


# In[79]:


#print X_train and X_test
print("X_train:",x_train)
print("X_test:",x_test)


# In[80]:


#print the shape of X_train and X_test
print("Shape of X_train",x_train.shape)
print("Shape of X_test",x_test.shape)


# In[81]:


#print the shape of Y_train and Y_test
print("Shape of Y_train",Y_train.shape)
print("Shape of Y_test",Y_test.shape)


# In[82]:


#train the model
rf= RandomForestRegressor(n_estimators=99,max_depth=10,random_state=99)
model=rf.fit(x_train,Y_train)


# In[84]:


#test the model 
Y_predict_Rf=model.predict(x_test)


# In[86]:


#CALCULATING REGRESSION METRICS
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
mae_RandomForest=mean_absolute_error(Y_predict_Rf,Y_test)
mse_RandomForest=mean_squared_error(Y_predict_Rf,Y_test)
r2_RandomForest=r2_score(Y_test, Y_predict_Rf)


# In[87]:


print("mean_absolute_error of RandomForestRegressor",mean_absolute_error(Y_predict_Rf,Y_test))
print("mean_squared_error of RandomForestRegressor:",mean_squared_error(Y_predict_Rf,Y_test))
print("R2 value of RandomForestRegressor is:", r2_score(Y_test, Y_predict_Rf))


# In[88]:


#3.SupportVectorRegressor
#import neccessary libaries
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split


# In[89]:


#Assigning values for independent and dependent varaibles
x_vars = ["R&D Spend","Administration","Marketing Spend"]
X = df[x_vars]
Y=Y


# In[90]:


#train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(x, Y, test_size=0.25, random_state=99)


# In[91]:


#print the Shape of X and Y
print("Shape of X:",X.shape)
print("Shape of Y:",Y.shape)


# In[92]:


#print the shape of X_train and X_test
print("Shape of X_train",X_train.shape)
print("Shape of X_test",X_test.shape)


# In[93]:


#print Y_train and Y_test
print("Y_train:",Y_train)
print("Y_test:",Y_test)


# In[95]:


#train the model
svr=SVR()
Regressor=svr.fit(X_train,Y_train)


# In[96]:


#test the model 
Y_predict_SVR=Regressor.predict(X_test)


# In[97]:


#CALCULATING REGRESSION METRICS
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


# In[99]:


mae_SupportVector=mean_absolute_error(Y_predict_SVR,Y_test)
mse_SupportVector=mean_squared_error(Y_predict_SVR,Y_test)
r2_SupportVector=r2_score(Y_test, Y_predict_SVR)


# In[101]:


print("mean_absolute_error of SupportVectorRegressor:",mean_absolute_error(Y_predict_SVR,Y_test))
print("mean_squared_error of SupportVectorRegressor:",mean_squared_error(Y_predict_SVR,Y_test))
print("R2 value of SupportVectorRegressor is:", r2_score(Y_test, Y_predict_SVR))


# In[102]:


#4.Ridge Regression
#import all the necessary libararies
from sklearn.linear_model import Ridge 


# In[103]:


#Assigning values for independent and dependent varaibles
x_vars = ["R&D Spend","Administration","Marketing Spend"]
X = df[x_vars]
Y=Y


# In[104]:


#train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(x, Y, test_size=0.25, random_state=99)


# In[105]:


#print the Shape of X and Y
print("Shape of X:",X.shape)
print("Shape of Y:",Y.shape)


# In[106]:


#print the shape of X_train and X_test
print("Shape of X_train",X_train.shape)
print("Shape of X_test",X_test.shape)


# In[107]:


#print Y_train and Y_test
print("Y_train:",Y_train)
print("Y_test:",Y_test)


# In[108]:


#Fit the model
ridge =  Ridge()
Redige_regressor=ridge.fit(X_train,Y_train)
    


# In[109]:


#test the model
Y_pred_ridge = Redige_regressor.predict(X_test)


# In[112]:


#CALCULATING REGRESSION METRICS
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
mae_RidgeRegressor=mean_absolute_error(Y_pred_ridge,Y_test)
mse_RidgeRegressor=mean_squared_error(Y_pred_ridge,Y_test)
r2_RidgeRegressor=r2_score(Y_test, Y_pred_ridge)


# In[114]:


print("mean_absolute_error of Ridge Regression is:",mean_absolute_error(Y_pred_ridge,Y_test))
print("mean_squared_error of Ridge Regression is:",mean_squared_error(Y_pred_ridge,Y_test))
print("R2 value of SupportVectorRegressor is:", r2_score(Y_test, Y_pred_ridge))


# In[115]:


#5.Lasso Regression
#import libraries
from sklearn.linear_model import Lasso


# In[116]:


#Assigning values for independent and dependent varaibles
x_vars = ["R&D Spend","Administration","Marketing Spend"]
X = df[x_vars]
Y=Y


# In[117]:


#train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(x, Y, test_size=0.25, random_state=99)


# In[118]:


#print the Shape of X and Y
print("Shape of X:",X.shape)
print("Shape of Y:",Y.shape)


# In[119]:


#print the shape of X_train and X_test
print("Shape of X_train",X_train.shape)
print("Shape of X_test",X_test.shape)


# In[120]:


#print Y_train and Y_test
print("Y_train:",Y_train)
print("Y_test:",Y_test)


# In[121]:


#Fit the model
lassoreg = Lasso()
Lasso_reg=lassoreg.fit(X_train,Y_train)


# In[122]:


#test the model
Y_pred_lasso = Lasso_reg.predict(X_test)


# In[123]:


#CALCULATING REGRESSION METRICS
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
mae_LassoRegression=mean_absolute_error(Y_pred_lasso,Y_test)
mse_LassoRegression=mean_squared_error(Y_pred_lasso,Y_test)
r2_LassoRegression=r2_score(Y_test, Y_pred_lasso)


# In[124]:


print("Mean_absolute_error of Lasso Regression is:",mean_absolute_error(Y_pred_lasso,Y_test))
print("Mean_squared_error of Lasso Regression is:",mean_squared_error(Y_pred_lasso,Y_test))
print("R2 value of Lasso Regression is:", r2_score(Y_test, Y_pred_lasso))


# In[125]:


#6.Decision tree
#import neccesary libarires
from sklearn.tree import DecisionTreeRegressor


# In[126]:


#Assigning values for independent and dependent varaibles
x_vars = ["R&D Spend","Administration","Marketing Spend"]
X = df[x_vars]
Y=Y


# In[127]:


#train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(x, Y, test_size=0.25, random_state=99)


# In[128]:


#print the Shape of X and Y
print("Shape of X:",X.shape)
print("Shape of Y:",Y.shape)


# In[129]:


#print the shape of X_train and X_test
print("Shape of X_train",X_train.shape)
print("Shape of X_test",X_test.shape)


# In[130]:


#print Y_train and Y_test
print("Y_train:",Y_train)
print("Y_test:",Y_test)


# In[131]:


#fit the model
decision_tree_reg = DecisionTreeRegressor()
DecisionTree=decision_tree_reg.fit(X_train, Y_train)


# In[132]:


#test the model
Y_pred_Decisiontree = DecisionTree.predict(X_test)


# In[133]:


#CALCULATING REGRESSION METRICS
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
mae_DecisionTree=mean_absolute_error(Y_pred_Decisiontree,Y_test)
mse_DecisionTree=mean_squared_error(Y_pred_Decisiontree,Y_test)
r2_DecisionTree=r2_score(Y_test, Y_pred_Decisiontree)


# In[134]:


print("Mean_absolute_error of DescisionTree Regression is:",mean_absolute_error(Y_pred_Decisiontree,Y_test))
print("Mean_squared_error of DescisionTree Regression is:",mean_squared_error(Y_pred_Decisiontree,Y_test))
print("R2 value of DescisionTree Regression is:", r2_score(Y_test, Y_pred_Decisiontree))


# In[135]:


#7.GradientBoostingRegressor
#import neccesary libaries
from sklearn.ensemble import GradientBoostingRegressor


# In[136]:


#Assigning values for independent and dependent varaibles
x_vars = ["R&D Spend","Administration","Marketing Spend"]
X = df[x_vars]
Y=Y


# In[137]:


#train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(x, Y, test_size=0.25, random_state=99)


# In[138]:


#print the Shape of X and Y
print("Shape of X:",X.shape)
print("Shape of Y:",Y.shape)


# In[139]:


#print the shape of X_train and X_test
print("Shape of X_train",X_train.shape)
print("Shape of X_test",X_test.shape)


# In[140]:


#print Y_train and Y_test
print("Y_train:",Y_train)
print("Y_test:",Y_test)


# In[142]:


#fit the model
model = GradientBoostingRegressor()
model.fit(X_train, Y_train)


# In[147]:


#test the model
Y_pred_Gradient = model.predict(X_test)


# In[148]:


#CALCULATING REGRESSION METRICS
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
mae_GradientBoosting=mean_absolute_error(Y_pred_Gradient,Y_test)
mse_GradientBoosting=mean_squared_error(Y_pred_Gradient,Y_test)
r2_GradientBoosting=r2_score(Y_test, Y_pred_Gradient)


# In[149]:


print("Mean_absolute_error of GradientBoosting is:",mean_absolute_error(Y_pred_Gradient,Y_test))
print("Mean_squared_error of GradientBoosting is:",mean_squared_error(Y_pred_Gradient,Y_test))
print("R2 value of GradientBoosting is:", r2_score(Y_test, Y_pred_Gradient))


# In[219]:


# Plot the R-squared values for all the models
plt.figure(figsize=(10, 6))
plt.bar(models[:len(valid_r2_values)], valid_r2_values)
plt.xlabel('Model')
plt.ylabel('R-squared')
plt.title('R-squared Comparison for Different Regression Models')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# In[ ]:




