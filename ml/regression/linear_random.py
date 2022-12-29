import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error

# 데이터 준비
num = 51
x = np.linspace(0, 10, num)
y = x + np.random.normal(1, 2, num)

sns.scatterplot(x=x, y=y)

# 데이터 나누기
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=2022)

x_train = x_train.reshape(-1, 1)  # 2차원으로 데이터 형태 변경

# Fit
lr = LinearRegression()
lr.fit(x_train, y_train)

# 최종 선
# print(lr.coef_)
# print(lr.intercept_)

y_pred = lr.predict(x_train)
# print(y_pred)

sns.scatterplot(x=x_train.reshape(-1), y=y_train)
plt.plot(x_train.reshape(-1), y_pred, "r")
plt.show()

# 평가
print(lr.score(x_train, y_train))

# MSE
print(mean_squared_error(y_train, y_pred))
# RMSE
print(np.sqrt(mean_squared_error(y_train, y_pred)))
# MAE
print(mean_absolute_error(y_train, y_pred))

# 테스트
y_pred = lr.predict(x_test.reshape(-1, 1))
print(lr.score(x_test.reshape(-1, 1), y_test))

# RMSE
print(np.sqrt(mean_squared_error(y_test, y_pred)))

# 시각화
sns.scatterplot(x=y_test, y=y_pred)
plt.plot([0, 10], [0, 10], "r")
plt.show()
