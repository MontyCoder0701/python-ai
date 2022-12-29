import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

# Data
df = pd.read_csv("ml/logistic/diabetes.csv")

# 결측치, 중복치 확인
# print(df.isna().sum())
# print(df.duplicated().sum())

# 이상치 확인


def iszero(x):
    return x == 0


zeros = df.apply(iszero).sum(axis=0)
print(zeros)

print(df.loc[df["Glucose"] == 0])

# 이상치 평균으로 바꾸기
for col in ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]:
    df[col].replace(0, np.nan, inplace=True)
    df[col].fillna(df[col].mean(), inplace=True)

# x, y 지정
x = df.drop(['Outcome'], axis=1)
y = df['Outcome']

# 학습셋 분리
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=2022)

# 표준화
scaler = StandardScaler()
scaler.fit_transform(x_train)
y_train = y_train.values

# Fit
lr = LogisticRegression(max_iter=1000)
lr.fit(x_train, y_train)

# 최종
print(lr.score(x_train, y_train))

# 테스트
y_pred = lr.predict(x_test)
print(lr.score(x_test, y_pred))

y_pred = lr.predict_proba(x_test)
y_pred = np.argmax(y_pred, axis=1)
print(y_pred)


def plot_confusion_matrix(y_true, y_pred):
    cfm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(5, 5))
    sns.heatmap(cfm, annot=True, cbar=False, fmt="d")
    plt.xlabel("Predicted Class")
    plt.ylabel("True Class")
    plt.show()


plot_confusion_matrix(y_test, y_pred)
