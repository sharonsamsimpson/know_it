#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from flask import Flask, render_template, request
import pickle

app = Flask(__name__)
model_knn = pickle.load(open('model_knn.pkl', 'rb'))
model_lr = pickle.load(open('model_lr.pkl', 'rb'))
model_lor = pickle.load(open('model_lor.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/pizza', methods=['POST', 'GET'])
def rpizza():
    return render_template('pizza.html')

@app.route('/pizza.html', methods=['POST', 'GET'])
def pizza():
    float_features = [float(x) for x in request.form.values()]
    final_features = [np.array(float_features)]
    prediction = model_knn.predict(final_features)
    if prediction == 1:
        pred = "like Pizza"
    elif prediction == 0:
        pred = "don't like Pizza"
    output = pred
    return render_template('pizza.html', prediction_text='You {}'.format(output))

@app.route('/fuel', methods=['POST', 'GET'])
def rfuel():
    return render_template('fuel.html')

@app.route('/fuel.html', methods=['POST', 'GET'])
def fuel():
    float_features = [float(x) for x in request.form.values()]
    final_features = [np.array(float_features)]
    prediction = model_lr.predict(final_features)
    output = int(prediction[0])
    return render_template('fuel.html', prediction_text='Fuel price for kilometer driven is : {}'.format(output))

@app.route('/diabetes', methods=['POST', 'GET'])
def rdiabetes():
    return render_template('diabetes.html')

@app.route('/diabetes.html', methods=['POST', 'GET'])
def diabetes():
    float_features = [float(x) for x in request.form.values()]
    final_features = [np.array(float_features)]
    prediction = model_lor.predict(final_features)
    if prediction == 1:
        pred = "are likely to have DIABETES"
    elif prediction == 0:
        pred = "don't worry! You are not likely to have DIABETES"
    output = pred
    return render_template('diabetes.html', prediction_text='You {}'.format(output))

@app.route('/back', methods=['POST', 'GET'])
def back():
    return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True)

