import sys,codecs
import numpy as np
import pandas as pd
import sys

titanic=pd.read_csv('titanic.csv')
titanic=titanic.drop(['name','row.names'],axis=1)
mean=round(titanic['age'].mean(),2)
titanic['age'].fillna(mean,inplace=True)
titanic.fillna("",inplace=True)
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
for i in titanic.columns.values.tolist():
 if (i=='age'):
  pass
 else:
  titanic[i] = le.fit_transform(titanic[i])

filename='titanic.csv'
trees=493
crossv=10 
print('filename:',filename)
print("data instances and parameters:",titanic.shape)
print("cross-validation:",crossv)
print("trees:",trees)
y=titanic['survived']
X=titanic.drop(['survived'],axis=1)
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import *
from sklearn.model_selection import ShuffleSplit
cv = ShuffleSplit(n_splits=crossv, test_size=0.2, random_state=54) 
clf=RandomForestClassifier(criterion='entropy',n_estimators=trees, max_depth=None,min_samples_split=2, random_state=54,n_jobs=-1)
scores = cross_val_score(clf, X, y, cv=cv)
print(scores,scores.mean(),round(scores.std(),6))