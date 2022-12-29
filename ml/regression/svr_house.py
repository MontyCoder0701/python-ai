import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

# 데이터 준비
housing = fetch_california_housing()

# Df로 변환
df = pd.DataFrame(data=housing.data, columns=housing.feature_names)
df["target"] = housing.target

# x,y 분리
x = df[["MedInc", "HouseAge", "AveRooms"]]
y = df['target']

# 학습셋 분리
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=2022)

# 표준화
scaler = StandardScaler()
scaler.fit_transform(x_train)
y_train = y_train.values

# Fit
lr = SVR()
lr.fit(x_train, y_train)
print(lr.score(x_train, y_train))

# 테스트
y_pred = lr.predict(x_test)
print(lr.score(x_test, y_test))

# RMSE
print(np.sqrt(mean_squared_error(y_test, y_pred)))
# Cross validation
print(cross_val_score(lr, x_test, y_test, scoring="neg_mean_squared_error", cv=3))
