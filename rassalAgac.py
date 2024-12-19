# -*- coding: utf-8 -*-
"""
Created on Thu Dec 19 23:28:08 2024

@author: osmnb
"""

#kütüphanelerin import edilmesi
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# veri yükleme
veriler = pd.read_csv("maaslar.csv")

#data frame dilimleme(slice)
x = veriler.iloc[:,1:2]
y = veriler.iloc[:,2:]

#dilimleri numpy array'e çevirdik
X = x.values
Y = y.values

#lineer regresyon
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X,Y) #x kullanılarak y öğreniliyor



#polinomal regresyon
#doğrusal olmayan model oluşturma
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree=2)
x_poly = poly_reg.fit_transform(X)
print(x_poly)

lin_reg2 = LinearRegression()
lin_reg2.fit(x_poly, y)


#4. dereceden polinom
poly_reg = PolynomialFeatures(degree=4)
x_poly = poly_reg.fit_transform(X)
print(x_poly)

lin_reg2 = LinearRegression()
lin_reg2.fit(x_poly, y)

#görselleştirme
plt.scatter(X, Y,color="red")
plt.plot(X,lin_reg2.predict(poly_reg.fit_transform(X)))
plt.show()

plt.scatter(X,Y,color="red")
plt.plot(x,lin_reg.predict(X),color="blue")
plt.show()

plt.scatter(X, Y,color="red")
plt.plot(X,lin_reg2.predict(poly_reg.fit_transform(X)))
plt.show()

#tahminler

print(lin_reg.predict([[11]]))
print(lin_reg.predict([[6.6]]))

print(lin_reg2.predict(poly_reg.fit_transform([[11]])))
print(lin_reg2.predict(poly_reg.fit_transform([[6.6]])))


#verilerimizin standartlaştırılması(ölçeklendirme)
from sklearn.preprocessing import StandardScaler

sc1 = StandardScaler()
x_olcekli = sc1.fit_transform(X)
sc2 = StandardScaler()
y_olcekli = sc2.fit_transform(Y)


from sklearn.svm import SVR
svr_reg = SVR(kernel="rbf")
svr_reg.fit(x_olcekli, y_olcekli)

plt.scatter(x_olcekli,y_olcekli)
plt.plot(x_olcekli,svr_reg.predict(x_olcekli))
plt.show()
print(svr_reg.predict([[11]]))
print(svr_reg.predict([[6.6]]))


from sklearn.tree import DecisionTreeRegressor
r_dt = DecisionTreeRegressor(random_state=0)
r_dt.fit(X,Y)
Z = X + 0.5
K = X - 0.5
plt.scatter(X,Y)
plt.plot(X,r_dt.predict(X))

plt.plot(x, r_dt.predict(Z),color="red")
plt.plot(x, r_dt.predict(K),color="green")
plt.show()
print(r_dt.predict([[11]]))
print(r_dt.predict([[6.6]]))





from sklearn.ensemble import RandomForestRegressor
rf_reg = RandomForestRegressor(n_estimators=10,random_state=0)

rf_reg.fit(X,Y.ravel())
print(rf_reg.predict([[6.6]]))

plt.scatter(X, Y,color = "red")
plt.plot(X,rf_reg.predict(X),color="blue")

plt.plot(X,rf_reg.predict(Z),color="green")

plt.plot(X,rf_reg.predict(K),color="yellow")


















