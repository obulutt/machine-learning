# -*- coding: utf-8 -*-
"""
Created on Fri Dec  6 23:43:49 2024

@author: osmnb
"""

#kütüphanelerin import edilmesi
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# veri yükleme
veriler = pd.read_csv("odev_tenis.csv")

print(veriler)


#encoder: nominal veya ordinalden -> nümeriğe dönüştürme

from sklearn import preprocessing
veriler2 = veriler.apply(preprocessing.LabelEncoder().fit_transform)

c= veriler2.iloc[:,:1]


from sklearn import preprocessing
ohe = preprocessing.OneHotEncoder()
c = ohe.fit_transform(c).toarray()
print(c)


havaDurumu = pd.DataFrame(data=c,index=range(14),columns=["o","r","s"])
sonVeriler = pd.concat([havaDurumu,veriler.iloc[:,1:3]],axis=1)
sonVeriler = pd.concat([veriler2.iloc[:,-2:],sonVeriler],axis=1)




#verilerin eğitim ve test için bölünmesi
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(sonVeriler.iloc[:,:-1],sonVeriler.iloc[:,-1:],test_size=0.33,random_state=0)


from sklearn.linear_model import LinearRegression

regressor = LinearRegression()
regressor.fit(x_train,y_train)

y_pred = regressor.predict(x_test)




#geriye doğru eleme
import statsmodels.api as sm

X = np.append(arr=np.ones((14,1)).astype(int),values=sonVeriler.iloc[:,:-1],axis=1)

X_l = sonVeriler.iloc[:,[0,1,2,3,4,5]].values
X_l = np.array(X_l,dtype=float)
model = sm.OLS(sonVeriler.iloc[:,-1:],X_l).fit()

print(model.summary())

sonVeriler = sonVeriler.iloc[:,1:]

import statsmodels.api as sm
X = np.append(arr=np.ones((14,1)).astype(int),values=sonVeriler.iloc[:,:-1],axis=1)

X_l = sonVeriler.iloc[:,[0,1,2,3,4]].values
X_l = np.array(X_l,dtype=float)
model = sm.OLS(sonVeriler.iloc[:,-1:],X_l).fit()
print(model.summary())

x_train = x_train.iloc[:,1:]
x_test = x_test.iloc[:,1:]
regressor.fit(x_train,y_train)

y_pred = regressor.predict(x_test)









