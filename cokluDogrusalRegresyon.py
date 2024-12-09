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
veriler = pd.read_csv("veriler.csv")

print(veriler)
boy = veriler[["boy"]]
print(boy)

boyKilo = veriler[["boy","kilo"]]
print(boyKilo)

Yas = veriler.iloc[:,1:4].values
print(Yas)

#encoder: nominal veya ordinalden -> nümeriğe dönüştürme

ulke = veriler.iloc[:,0:1].values
print(ulke)

from sklearn import preprocessing

le = preprocessing.LabelEncoder()
ulke[:,0] = le.fit_transform(veriler.iloc[:,0])

print(ulke)

ohe = preprocessing.OneHotEncoder()

ulke = ohe.fit_transform(ulke).toarray()
print(ulke)



c = veriler.iloc[:,-1:].values
print(c)

from sklearn import preprocessing

le = preprocessing.LabelEncoder()
c[:,-1] = le.fit_transform(veriler.iloc[:,-1])

print(c)

ohe = preprocessing.OneHotEncoder()
c = ohe.fit_transform(c).toarray()
print(c)


#numpy dizilerinin df e dönüşümü
sonuc = pd.DataFrame(data=ulke,index=range(22),columns=["fr","tr","us"])
print(sonuc)

sonuc2=pd.DataFrame(data=Yas,index=range(22),columns=["boy","kilo","yas"])
print(sonuc2)

cinsiyet = veriler.iloc[:,-1]
print(cinsiyet)

sonuc3 = pd.DataFrame(data=c[:,:1],index=range(22),columns=["cinsiyet"])
print(sonuc3)

#df birleştirme işlemleri
s = pd.concat([sonuc,sonuc2],axis=1)
print(s)

s2 = pd.concat([s,sonuc3],axis=1)
print(s2)

#verilerin eğitim ve test için bölünmesi
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(s,sonuc3,test_size=0.33,random_state=0)


from sklearn.linear_model import LinearRegression

regressor = LinearRegression()
regressor.fit(x_train,y_train)

y_pred = regressor.predict(x_test)


sol = s2.iloc[:,:3]
sag = s2.iloc[:,4:]

veri = pd.concat([sol,sag],axis=1)

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(veri,boy,test_size=0.33,random_state=0)


r2 = LinearRegression()
r2.fit(x_train,y_train)

y_pred = r2.predict(x_test)

"""
#geriye doğru eleme
import statsmodels.api as sm

X = np.append(arr=np.ones((22,1)).astype(int),values=veri,axis=1)

X_l = veri.iloc[:,[0,1,2,3,4,5]].values
X_l = np.array(X_l,dtype=float)
model = sm.OLS(boy,X_l).fit()

print(model.summary())


#çıkan sonuca göre 4 numaralı olan kolonu eliyoruz
# sonra 5 numaraları kolonu da eliyoruz
"""

#geriye doğru eleme
import statsmodels.api as sm

X = np.append(arr=np.ones((22,1)).astype(int),values=veri,axis=1)

X_l = veri.iloc[:,[0,1,2,3]].values
X_l = np.array(X_l,dtype=float)
model = sm.OLS(boy,X_l).fit()

print(model.summary())














