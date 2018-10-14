#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 13 13:50:15 2018

@author: stille
"""

import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn import cross_validation
from sklearn.metrics import mean_squared_error as MSE
from sklearn.cross_validation import KFold, cross_val_score
from sklearn.metrics import accuracy_score

df_wine = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data',header=None)
df_wine.columns = ['Class label', 'Alcohol','Malic acid', 'Ash','Alcalinity of ash', \
                   'Magnesium', 'Total phenols','Flavanoids', 'Nonflavanoid phenols', \
                   'Proanthocyanins','Color intensity', 'Hue','OD280/OD315 of diluted wines','Proline']
cols=['Class label', 'Alcohol','Malic acid', 'Ash','Alcalinity of ash', \
                   'Magnesium', 'Total phenols','Flavanoids', 'Nonflavanoid phenols', \
                   'Proanthocyanins','Color intensity', 'Hue','OD280/OD315 of diluted wines','Proline']
df_wine = df_wine[df_wine['Class label'] != 1]
y = df_wine['Class label'].values
X = df_wine[['Alcohol', 'Hue']].values

X, y = df_wine.iloc[:, 1:].values, df_wine.iloc[:, 0].values

X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.1, random_state=80)

for num in range(1,20):
    print("n estimators = {}".format(num))
    rf = RandomForestClassifier(n_estimators=num,random_state=0,n_jobs=-1)
    # Fit rf to the training set    
    rf.fit(X_train,y_train) 
    # Predict the test set labels
    y_pred = rf.predict(X_test)
    
    
    k_fold = KFold(len(y_train), n_folds=10, shuffle=True, random_state=0)
    print("    In sample accuracy:")
    print("        ",np.mean(cross_val_score(rf, X_train, y_train, cv=k_fold, n_jobs=1)))
    print("    Out of sample accuracy:")
    print("        ",accuracy_score(y_test,y_pred))

rf = RandomForestClassifier(n_estimators=15,random_state=0,n_jobs=-1)

rf.fit(X,y) 

feat_labels = df_wine.columns[1:]

importances = rf.feature_importances_

indices = np.argsort(importances)[::-1]

for f in range(X_train.shape[1]):
        print("%2d) %-*s %f" % (f + 1, 30,feat_labels[f],importances[indices[f]]))
    


print("My name is {Wenyu Ni}")
print("My NetID is: {wenyuni2}")
print("I hereby certify that I have read the University policy on Academic Integrity and that I am not in violation.")


