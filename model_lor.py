#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pickle


# In[2]:


df = pd.read_csv("diabetes.csv")
X = df.drop('Outcome', axis=1)
y = df['Outcome'].values


# In[3]:


X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.25, random_state=42)


# In[4]:


lor = LogisticRegression(penalty='l2', C=10.0)
lor.fit(X_train, y_train)
lor.predict(X_test)


# In[5]:


pickle.dump((lor), open('model_lor.pkl','wb'))
model_lor = pickle.load(open('model_lor.pkl','rb'))


# In[ ]:




