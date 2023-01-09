# 2 class data를 다중분류로 처리 (one hot)

from keras.utils import to_categorical
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from keras import layers
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("ml/logistic/diabetes.csv")

# x, y 지정
x = df.drop(['Outcome'], axis=1)
y = df['Outcome']

# 학습셋 분리
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2)

# 정규화
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
y_train = y_train.values  # numpy로 변환

x_test = scaler.transform(x_test)  # fit_transform이 아님에 유의

# 이중분류
# # 모델
# model = keras.Sequential([
#     layers.Dense(units=16, activation="relu",
#                  input_shape=(8,)),  # input_dim은 dense인 경우에만
#     layers.Dense(units=8, activation="relu"),
#     # 이진 분류 모델 (sigmoid 출력층 - 0 혹은 1)/ 다중분류 (softmax)
#     layers.Dense(units=1, activation="sigmoid"),
# ])

# # Compile

# model.compile(
#     optimizer="adam",
#     loss="binary_crossentropy",
#     metrics=["acc"]
# )

# model.summary()

# # 학습
# epochs = 30
# batch_size = 32

# history = model.fit(x_train, y_train, epochs=epochs,
#                     batch_size=batch_size, validation_split=0.2, verbose=1)

# # 평가
# model.evaluate(x_train, y_train)
# model.evaluate(x_test, y_test)

# # 예측
# y_test = y_test.values
# y_pred = model.predict(x_test)
# print(y_pred)

# # 0과 1로 변환 (이진분류)- Sigmoid 결과값
# y_pred = y_pred.reshape(-1)
# y_pred = ((y_pred > 0.5).astype(int))
# print(y_pred)

# 다중분류
y_train_oh = to_categorical(y_train)
y_test_oh = to_categorical(y_test)

# 모델
model = keras.Sequential([
    layers.Dense(units=16, activation="relu",
                 input_shape=(8,)),  # input_dim은 dense인 경우에만
    layers.Dense(units=8, activation="relu"),
    # 이진 분류 모델 (sigmoid 출력층 - 0 혹은 1)/ 다중분류 (softmax)
    layers.Dense(units=2, activation="softmax"),
])

# Compile

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["acc"]
)

# 학습
epochs = 30
batch_size = 32

history = model.fit(x_train, y_train_oh, epochs=epochs,
                    batch_size=batch_size, validation_split=0.2, verbose=1)

# 평가
model.evaluate(x_train, y_train_oh)
model.evaluate(x_test, y_test_oh)

# 예측
y_pred = model.predict(x_test)
print(y_pred)

# 다중분류
y_pred = np.argmax(y_pred, axis=1)
print(y_pred)
