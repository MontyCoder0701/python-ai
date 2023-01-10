from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from keras import layers
import pandas as pd

# 이상치 처리
df = pd.read_csv("tensorflow/linear/auto-mpg.csv", na_values=["?"])
print(df)

# See data info
# print(df.isna().sum())
# print(df.duplicated().sum())
# print(df.describe().T)

# 결측치 제거
df = df.dropna()

x = df.drop(["mpg", "origin", "car name"], axis=1)
y = df["mpg"]


# 테스트셋 나누기
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=2022)

# 정규화
scaler = StandardScaler()
x_train_s = scaler.fit_transform(x_train)
y_train = y_train.values

# 모델
model = keras.Sequential([
    layers.Dense(64, activation="relu", input_shape=(6,)),
    layers.Dense(32, activation="relu"),
    layers.Dense(1)
])

model.summary()

# Compile
model.compile(
    optimizer="adam",
    loss="mse",
    metrics=["mse", "mae"]
)

# 학습
epochs = 200
batch_size = 32

history = model.fit(x_train_s, y_train, epochs=epochs,
                    batch_size=batch_size, validation_split=0.2, verbose=1)

# 평가
model.evaluate(x_train_s, y_train)

# 로그 확인
print(history.history.keys())

# matplotlib


def plot_history(history):
    hist = pd.DataFrame(history.history)
    hist["epoch"] = history.epoch
    plt.plot(hist["epoch"], hist["mse"], label="Train MSE")
    plt.plot(hist["epoch"], hist["val_mse"], label="Val MSE")
    plt.legend()
    plt.show()


plot_history(history)

# 예측
x_test_s = scaler.transform(x_test)
y_test = y_test.values

y_pred = model.predict(x_test_s)

# 모양 맞춰주기
print(y_pred.shape)
print(y_test.shape)

y_pred = y_pred.reshape(-1)
print(mean_squared_error(y_test, y_pred))
