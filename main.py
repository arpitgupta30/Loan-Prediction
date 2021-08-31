import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#import the dataset
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

#print the datatypes
train.dtypes
test.dtypes

#print the columns
train.columns
test.columns

#the dimensions of dataset
train.shape
test.shape

#make the copy of the dataset
train_copy = train.copy()
test_copy = test.copy()

#univariate analysis

#for all categorical data 
train['Loan_Status'].value_counts(normalize = True).plot.bar()
train['Loan_Status'].value_counts().plot.bar(title = 'Loan Status')

#for gender variable
train['Gender'].value_counts().plot.bar(title = 'Gender')

#for married variable
train['Married'].value_counts().plot.bar(title = 'Marital Status')

#for dependents
train['Dependents'].value_counts().plot.bar(title = 'Dependents')

#for education
train['Education'].value_counts().plot.bar(title = 'Education')

#for employment
train['Self_Employed'].value_counts().plot.bar(title = 'Self Employed')

#for Credit_History
train['Credit_History'].value_counts().plot.bar(title = 'Credit_History')

#for property area
train['Property_Area'].value_counts().plot.bar(title = 'Property_Area')

#all the above features were categorical dimension

#for continuous variable

#for applicant income
plt.hist(train['ApplicantIncome'], edgecolor = 'black')
plt.title('Applicant Income')
plt.show()

#for co - applicant income
plt.hist(train['CoapplicantIncome'], edgecolor = 'black')
plt.title('Coapplicant Income')
plt.show()

#for loanAmount
#loan amount consist of null values so to construct it we need to drop those columns
df = train.dropna()
plt.hist(df['LoanAmount'], edgecolor = 'black')
plt.title('Loan Amount')
plt.show()

train.columns

#bivariate analysis

# for cross tab function analysis https://pbpython.com/pandas-crosstab.html
Gender=pd.crosstab(train['Gender'],train['Loan_Status']) 
# Gender.sum(1) = rowsum
Gender.div(Gender.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True, figsize=(4,4))

Married=pd.crosstab(train['Married'],train['Loan_Status'])
Married.div(Married.sum(1).astype(float), axis=0).plot(kind='bar', stacked=True)

Dependents = pd.crosstab(train['Dependents'], train['Loan_Status'])
Dependents.div(Dependents.sum(1).astype(float),axis=0).plot(kind = "bar",stacked=True)

Education = pd.crosstab(train['Education'],train['Loan_Status'])
Education.div(Education.sum(1).astype(float),axis=0).plot(kind = 'bar', stacked = True)

Self_Employed = pd.crosstab(train['Self_Employed'], train['Loan_Status'])
Self_Employed.div(Self_Employed.sum(1).astype(float), axis = 0).plot(kind = "bar", stacked=True)

Credit_History = pd.crosstab(train['Credit_History'], train['Loan_Status'])
Credit_History.div(Credit_History.sum(1).astype(float),axis=0).plot(kind="bar", stacked = True)

train.columns

Property_Area = pd.crosstab(train['Property_Area'], train['Loan_Status'])
Property_Area.div(Property_Area.sum(1).astype(float), axis = 0).plot(kind="bar", stacked=True)

#another way to construct the same graph
Property_Area.div(Property_Area.sum(1).astype(float), axis = 0).plot.bar(stacked = True)

train.groupby('Loan_Status')['ApplicantIncome'].mean().plot.bar()


#Important
#this is done inorder to create categories
#application income
bins=[0,2500,4000,6000,81000]
group=['Low','Average','High', 'Very high'] 
train['Income_bin']=pd.cut(df['ApplicantIncome'],bins,labels=group)

Income_bin = pd.crosstab(train['Income_bin'],train['Loan_Status'])
Income_bin.div(Income_bin.sum(1).astype(float),axis=0).plot.bar(stacked = True)

#coapplicant income
bins=[0,1000,3000,42000] 
group=['Low','Average','High'] 
train['Coapplicant_Income_bin']=pd.cut(df['CoapplicantIncome'],bins,labels=group)

Coapplicant_Income_bin = pd.crosstab(train['Coapplicant_Income_bin'],train['Loan_Status'])
Coapplicant_Income_bin.div(Coapplicant_Income_bin.sum(1).astype(float),axis=0).plot.bar(stacked=True)

#total income
train['Total_Income'] = train['ApplicantIncome']+train['CoapplicantIncome']
bins=[0,2500,4000,6000,81000]
group=['Low','Average','High', 'Very high']
train['Total_Income_bin']=pd.cut(train['Total_Income'],bins,labels=group)

Total_Income_bin = pd.crosstab(train['Total_Income_bin'], train['Loan_Status'])
Total_Income_bin.div(Total_Income_bin.sum(1).astype(float), axis=0).plot.bar(stacked=True)

#loan amount
bins=[0,100,200,700] 
group=['Low','Average','High']
train['LoanAmount_bin'] = pd.cut(train['LoanAmount'],bins,labels = group)

LoanAmount_bin = pd.crosstab(train['LoanAmount_bin'],train['Loan_Status'])
LoanAmount_bin.div(LoanAmount_bin.sum(1).astype(float),axis=0).plot.bar(stacked=True)

#dropping the columns that were previously added
train = train.drop(['LoanAmount_bin','Total_Income_bin','Coapplicant_Income_bin','Income_bin'], axis=1)
train = train.drop(['Total_Income'], axis=1)


#to handle missing values
train.isnull().sum()

train['Gender'].value_counts()
mod = train['Gender'].mode()[0]
train['Gender'].fillna(mod,inplace = True)
test['Gender'].fillna(mod,inplace = True)

train['Married'].value_counts()
mod = train['Married'].mode()[0]
train['Married'].fillna(mod,inplace = True)
test['Married'].fillna(mod,inplace = True)

train['Dependents'].value_counts()
mod = train['Dependents'].mode()[0]
train['Dependents'].fillna(mod,inplace = True)
test['Dependents'].fillna(mod, inplace = True)

train['Self_Employed'].value_counts()
mod = train['Self_Employed'].mode()[0]
train['Self_Employed'].fillna(mod, inplace = True)
test['Self_Employed'].fillna(mod, inplace = True)

train['Loan_Amount_Term'].value_counts()
mod = train['Loan_Amount_Term'].mode()[0]
train['Loan_Amount_Term'].fillna(mod, inplace = True)
test['Loan_Amount_Term'].fillna(mod, inplace = True)


train['Credit_History'].value_counts()
mod = train['Credit_History'].mode()[0]
train['Credit_History'].fillna(mod, inplace = True)
test['Credit_History'].fillna(mod, inplace = True)

train.isnull().sum()
test.isnull().sum()

import seaborn as sns
sns.boxplot(train['LoanAmount'])
sns.boxplot(test['LoanAmount'])

median = train['LoanAmount'].median()
train['LoanAmount'].fillna(median, inplace = True)
train['LoanAmount'].isnull().sum()
test['LoanAmount'].fillna(median,inplace = True)
plt.hist(train['LoanAmount'], edgecolor = 'black')
sns.boxplot(train['LoanAmount'])
sns.boxplot(test['LoanAmount'])

#take the log tranformation inorder to reduce the skewness of the dataset as there are presence of extrenious values
train['Log_LoanAmount'] = np.log(train['LoanAmount']) 
test['Log_LoanAmount'] = np.log(test['LoanAmount'])
plt.hist(train['Log_LoanAmount'])
sns.boxplot(train['Log_LoanAmount'])
sns.boxplot(test['Log_LoanAmount'])
plt.hist(test['Log_LoanAmount'])

train.head
train['Dependents'].replace('3+','3',inplace = True)
train.dtypes
test['Dependents'].replace('3+','3', inplace = True)

train['Total_Income'] = train['ApplicantIncome']+train['CoapplicantIncome']
plt.hist(train['Total_Income'])
sns.boxplot(train['Total_Income'])
train['Log_totalIncome'] = np.log(train['Total_Income'])
plt.hist(train['Log_totalIncome'])
sns.boxplot(train['Log_totalIncome'])

test['Total_Income'] = test['ApplicantIncome']+test['CoapplicantIncome']
plt.hist(test['Total_Income'])
sns.boxplot(test['Total_Income'])
test['Log_totalIncome'] = np.log(test['Total_Income'])
plt.hist(test['Log_totalIncome'])
sns.boxplot(test['Log_totalIncome'])

#taking care of categorical data
train = train.drop(['Loan_ID'], axis = 1)
train = pd.get_dummies(train)
test = test.drop(['Loan_ID'], axis = 1)
test = pd.get_dummies(test)

X = train.drop(['Loan_Status_N','Loan_Status_Y'], axis = 1)
Y = train['Loan_Status_Y']
X = X.drop(['ApplicantIncome', 'CoapplicantIncome', 'Total_Income'], axis = 1)
X_testDataset = test.drop(['ApplicantIncome', 'CoapplicantIncome', 'Total_Income'], axis = 1)


#training the data
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size = 0.3)


#using logistic regression
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state=0)
classifier.fit(X_train,Y_train)
#accuracy = 77.7%

#using random forest classifier
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(criterion = "entropy", n_estimators=10)
classifier.fit(X_train, Y_train)
#accuracy = 76%

Y_predict = classifier.predict(X_test)

from sklearn.metrics import confusion_matrix
confusion_matrix(Y_test, Y_predict)

#predictiing values
Y_testDataset = classifier.predict(X_testDataset)

#creating submission file
submission = pd.read_csv('Sample.csv')
submission['Loan_ID'] = test_copy['Loan_ID']
submission['Loan_Status'] = Y_testDataset
submission['Loan_Status'].replace(0,'N', inplace = True)
submission['Loan_Status'].replace(1,'Y', inplace = True)

#converting the file to csv
submission.to_csv('Output.csv', index = False)