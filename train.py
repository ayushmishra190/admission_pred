
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pickle
from sklearn.metrics import  r2_score
df=pd.read_csv(r'C:\Users\acer\Downloads\Projects\LinearRegression-master\LinearRegressionTillCloud/Admission_Prediction.csv')
df['GRE Score'].fillna(df['GRE Score'].mode()[0],inplace=True)
df['TOEFL Score'].fillna(df['TOEFL Score'].mode()[0],inplace=True)
df['University Rating'].fillna(df['University Rating'].mean(),inplace=True)
x=df.drop(['Chance of Admit','Serial No.'],axis=1)
y=df['Chance of Admit']
train_x,test_x,train_y,test_y=train_test_split(x,y,test_size=0.33, random_state=100)
reg = LinearRegression()
reg.fit(train_x, train_y)
core= r2_score(reg.predict(test_x),test_y)
filename = 'finalized_model.pickle'
pickle.dump(reg, open(filename, 'wb'))


