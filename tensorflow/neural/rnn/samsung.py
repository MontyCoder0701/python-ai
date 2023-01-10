# 삼성 주가 분석
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import keras
from keras import layers

# 데이터 준비
# 날짜 처리
df = pd.read_csv("tensorflow/neural/rnn/005930.KS.csv")
df["Date"] = pd.to_datetime(df["Date"])
df = df.set_index("Date")

# 시계열
# plt.figure(figsize=(10, 5))
# plt.plot(df.index, df["Close"])
# plt.show()

# Feature 추가
df["MA3"] = np.around(df["Close"].rolling(window=3).mean(), 0)  # 3일 평균선
df["MA5"] = np.around(df["Close"].rolling(window=5).mean(), 0)
df["Mid"] = (df["High"] + df["Low"])/2
print(df)

# 시계열
# x = df.iloc[-100:, :]  # 뒤에서 100개

# plt.figure(figsize=(10, 5))
# plt.plot(x.index, x["Close"])
# plt.plot(x.index, x["MA3"])
# plt.plot(x.index, x["MA5"])
# plt.plot(x.index, x["Mid"])
# # plt.show()

# 결손치 제거
print(df.isna().sum(axis=0))  # 결측치
print(df.loc[df["Volume"] == 0])  # 거래량이 0인 날

df["Volume"] = df["Volume"].replace(0, np.nan)
print(df.isna().sum(axis=0))  # 결측치

df = df.dropna()
print(df.isna().sum())

# 정규화
scaler = MinMaxScaler()
df_scaled = scaler.fit_transform(df)  # scaling (numpy)
df_scaled = pd.DataFrame(df_scaled, columns=df.columns)  # 다시 pd dataframe 형태

# RNN 예측 함수


def make_sequence_dataset(x, y, window_size):
    feature_list = []
    label_list = []

    for i in range(len(x) - window_size):
        feature_list.append(x[i: i+window_size])
        label_list.append(y[i+window_size])

    return np.array(feature_list), np.array(label_list)  # RNN에 맞는 형태 준비


x = df_scaled.drop(["Close", "Adj Close"], axis=1)
y = df_scaled["Close"]

x_data, y_data = make_sequence_dataset(x, y, 20)
print(x_data.shape, y_data.shape)


# Test train split
train_size = int(len(x_data) * 0.8)
x_train = x_data[0:train_size]
x_test = x_data[train_size:]

y_train = y_data[0:train_size]
y_test = y_data[train_size:]

print(x_train.shape, x_test.shape)

# RNN Model
model = keras.Sequential()
# RNN input shape (timestamp, features)
model.add(layers.LSTM(32, activation="tanh",
          return_sequences=True, input_shape=(20, 7)))
model.add(layers.Dropout(0.2))
model.add(layers.LSTM(32, activation="tanh"))
model.add(layers.Dropout(0.2))
model.add(layers.Dense(1))

model.summary()

# Compile
model.compile(
    optimizer="rmsprop",
    loss="mse",
    metrics=["mae"]
)

# 학습
epochs = 50
batch_size = 32

history = model.fit(x_train, y_train, epochs=epochs,
                    batch_size=batch_size, validation_split=0.2, verbose=1)

# matplotlib


def plot_history(history):
    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch

    plt.figure(figsize=(16, 8))
    plt.subplot(1, 2, 1)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.plot(hist['epoch'], hist['loss'], label='Train Loss')
    plt.plot(hist['epoch'], hist['val_loss'], label='Val Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.plot(hist['epoch'], hist['mae'], label='Train MAE')
    plt.plot(hist['epoch'], hist['val_mae'], label='Val MAE')
    plt.legend()

    plt.show()


plot_history(history)

# 평가
model.evaluate(x_train, y_train)

y_pred = model.predict(x_test)
y_pred = y_pred.reshape(-1)

for i in range(10):
    print("True", y_test[i], "Pred", y_pred[i])

plt.figure(figsize=(10, 5))
plt.plot(y_test[20:], label="true")
plt.plot(y_pred[20:], label="true")
plt.show()
