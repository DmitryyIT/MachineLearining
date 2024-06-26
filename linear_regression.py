# -*- coding: utf-8 -*-
"""Untitled1.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1lm3Q86dyoyDQkeGCiEi9fSboew-DqF-I
"""

prices = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190]
demand = [180, 170, 150, 140, 135, 128, 120, 115, 114, 108, 100, 97, 96, 95, 90, 85, 80, 80, 80]


import numpy as np


prices = np.array(prices).reshape(-1, 1)
demand = np.array(demand)


from sklearn.model_selection import train_test_split


X_train, X_test, y_train, y_test = train_test_split(prices, demand, test_size = 0.2, random_state = 42)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)


from sklearn.linear_model import LinearRegression


model = LinearRegression()
model.fit(X_train, y_train)
print(model.coef_, model.intercept_)
y_pred = model.predict(X_test)
y_pred[:3]


from sklearn import metrics


print('Root Mean Squared Error (RMSE):', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
