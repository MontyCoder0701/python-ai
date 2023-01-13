# RNN으로 이미지 분류하기
from keras.preprocessing.text import Tokenizer
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import keras
from keras import layers
from keras.datasets import mnist
from keras.utils import to_categorical

# 데이터 준비
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# 정규화
# Min-max 정규화 (0~1 사이 값)
x_train = x_train.astype("float32")/255
x_test = x_test.astype("float32")/255

# One-hot 이용하기
# Y값 One-hot 인코딩 (다중분류)
y_train_oh = to_categorical(y_train)
y_test_oh = to_categorical(y_test)

print(x_train.shape)  # (28,28)

# 모델 생성
model = keras.Sequential([
    layers.LSTM(64, activation="tanh", input_shape=(28, 28)),
    layers.Dense(units=10, activation="softmax")
])

model.compile(
    optimizer="rmsprop",  # RNN
    loss="categorical_crossentropy",
    metrics=["acc"]  # 분류의 경우 acc
)

model.summary()

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

# One-hot 전으로 되돌리기 (정답값의 형태로)
y_pred = np.argmax(y_pred, axis=1)

# 모델 저장
model.save("my_rnn_model.h5")
model.save("my_model")  # tensorflow
