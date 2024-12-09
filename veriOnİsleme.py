# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 13:58:22 2024

@author: osmnb
"""

#kütüphanelerin import edilmesi
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# veri yükleme
veriler = pd.read_csv("eksikveriler.csv")

print(veriler)

#veri ön işleme

boy = veriler[["boy"]]
print(boy)

boyKilo = veriler[["boy","kilo"]]
print(boyKilo)

#eksik veriler
#sci-kit learn
#♫eksik verilerin impute edilmesi

from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan,strategy="mean")

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

#numoy dizilerinin df e dönüşümü
sonuc = pd.DataFrame(data=ulke,index=range(22),columns=["fr","tr","us"])
print(sonuc)

sonuc2=pd.DataFrame(data=Yas,index=range(22),columns=["boy","kilo","yas"])
print(sonuc2)

cinsiyet = veriler.iloc[:,-1]
print(cinsiyet)

sonuc3 = pd.DataFrame(data=cinsiyet,index=range(22),columns=["cinsiyet"])
print(sonuc3)

#df birleştirme işlemleri
s = pd.concat([sonuc,sonuc2],axis=1)
print(s)

s2 = pd.concat([s,sonuc3],axis=1)
print(s2)

#verilerin eğitim ve test için bölünmesi
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(s,sonuc3,test_size=0.33,random_state=0)

#verilerimizin standartlaştırılması(ölçeklendirme)
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()

x_train = sc.fit_transform(x_train)
x_test = sc.fit_transform(x_test)














