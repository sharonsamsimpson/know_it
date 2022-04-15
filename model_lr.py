#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pickle

df2 = pd.read_csv("fuel.csv")
X2 = df2.filter(['drivenKM'])
y2 = df2['fuelAmount']

X_train, X_test, y_train, y_test = train_test_split(X2, y2, test_size=0.2, random_state=42)

lr = LinearRegression()
lr.fit(X_train, y_train)
lr.predict(X_test)

pickle.dump((lr), open('model_lr.pkl','wb'))
model_lr = pickle.load(open('model_lr.pkl','rb'))

