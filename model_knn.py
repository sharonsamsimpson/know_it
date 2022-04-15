#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
import pickle

df1 = pd.read_csv("pizza.csv")
X = df1.drop(['likePizza'], axis=1)
y = df1['likePizza']

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X, y)
knn.predict(X)

pickle.dump((knn), open('model_knn.pkl','wb'))
model_knn = pickle.load(open('model_knn.pkl','rb'))

