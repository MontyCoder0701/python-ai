# 다중분류를 one hot 없이 처리

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
from keras.datasets import mnist

# Data 준비
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# 시각화
samples = np.random.randint(60000, size=9)
for i, idx in enumerate(samples):
    plt.subplot(3, 3, i+1)
    plt.imshow(x_train[idx], cmap="gray")
    plt.xticks([])
    plt.yticks([])
    plt.title(y_train[idx])
# plt.show()

# 검증용 데이터
x_train, x_val, y_train, y_val = train_test_split(
    x_train, y_train, test_size=0.2)

# Min-max 정규화 (0~1 사이 값)
x_train = x_train.astype("float32")/255
x_val = x_val.astype("float32")/255
x_text = x_test.astype("float32")/255

# # One-hot 이용하기
# # Y값 One-hot 인코딩 (다중분류)
# y_train_oh = to_categorical(y_train)
# y_val_oh = to_categorical(y_val)
# y_test_oh = to_categorical(y_test)


# # 3차원으로 reshape해서 CNN에 넣기
# print(x_train.shape)

# x_train = x_train.reshape(-1, 28, 28, 1)
# x_val = x_val.reshape(-1, 28, 28, 1)
# x_test = x_test.reshape(-1, 28, 28, 1)

# # 모델 생성
# model = keras.Sequential([
#     layers.Conv2D(filters=32, kernel_size=3,
#                   activation="relu", input_shape=(28, 28, 1)),
#     layers.MaxPool2D(pool_size=(2, 2)),
#     layers.Conv2D(filters=64, kernel_size=3,
#                   activation="relu"),
#     layers.MaxPool2D(pool_size=(2, 2)),
#     # Dense 넣기 전에 flatten
#     layers.Flatten(),

#     layers.Dense(units=256, activation="relu"),
#     # 이진 분류 모델 (sigmoid 출력층 - 0 혹은 1)/ 다중분류 (softmax)
#     layers.Dense(units=10, activation="softmax"),
# ])

# model.compile(
#     optimizer="adam",
#     loss="categorical_crossentropy",
#     metrics=["acc"]  # 분류의 경우 acc
# )

# model.summary()

# # 학습
# epochs = 30
# batch_size = 32

# history = model.fit(x_train, y_train_oh, epochs=epochs,
#                     batch_size=batch_size, validation_data=(x_val, y_val_oh), verbose=1)


# # 평가
# model.evaluate(x_train, y_train_oh)
# model.evaluate(x_test, y_test_oh)

# # 예측
# y_pred = model.predict(x_test)

# # One-hot 전으로 되돌리기 (정답값의 형태로)
# y_pred = np.argmax(y_pred, axis=1)

# One hot 없이 처리

# 3차원으로 reshape해서 CNN에 넣기
print(x_train.shape)

x_train = x_train.reshape(-1, 28, 28, 1)
x_val = x_val.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)

# # 모델 생성
model = keras.Sequential([
    layers.Conv2D(filters=32, kernel_size=3,
                  activation="relu", input_shape=(28, 28, 1)),
    layers.MaxPool2D(pool_size=(2, 2)),
    layers.Conv2D(filters=64, kernel_size=3,
                  activation="relu"),
    layers.MaxPool2D(pool_size=(2, 2)),
    # Dense 넣기 전에 flatten
    layers.Flatten(),

    layers.Dense(units=256, activation="relu"),
    # 이진 분류 모델 (sigmoid 출력층 - 0 혹은 1)/ 다중분류 (softmax)
    layers.Dense(units=10, activation="softmax"),
])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",  # 자동으로 one-hot 처리
    metrics=["acc"]  # 분류의 경우 acc
)

# 학습
epochs = 30
batch_size = 32

history = model.fit(x_train, y_train, epochs=epochs,
                    batch_size=batch_size, validation_data=(x_val, y_val), verbose=1)

# 평가
model.evaluate(x_train, y_train)
model.evaluate(x_test, y_test)
# 예측
y_pred = model.predict(x_test)
print(y_pred)

# One-hot 전으로 되돌리기 (정답값의 형태로)
y_pred = np.argmax(y_pred, axis=1)
