# Binary classification (Diabetes dataset)
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, mean_squared_error
import tensorflow as tf
from tensorflow import keras
from keras import layers
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import seaborn as sns

df = pd.read_csv("ml/logistic/diabetes.csv")

# 결측치, 중복치 확인
print(df.isna().sum())
print(df.duplicated().sum())

# x, y 지정
x = df.drop(['Outcome'], axis=1)
y = df['Outcome']

# 학습셋 분리
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=2022)

# 표준화
scaler = StandardScaler()
x_train_s = scaler.fit_transform(x_train)
y_train = y_train.values

# 모델
model = keras.Sequential([
    layers.Dense(units=64, activation="relu", input_shape=(8,)),
    layers.Dense(units=32, activation="relu"),
    # 이진 분류 모델 (sigmoid 출력층 - 0 혹은 1)/ 다중분류 (softmax)
    layers.Dense(units=1, activation="sigmoid"),
])

model.summary()

# Compile

loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)

model.compile(
    optimizer="adam",
    loss=loss,
    metrics=["accuracy"]
)

# 학습
epochs = 50
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
    plt.ylabel('Accuracy')
    plt.plot(hist['epoch'], hist['accuracy'], label='Train Accuracy')
    plt.plot(hist['epoch'], hist['val_accuracy'], label='Val Accuracy')
    plt.legend()

    plt.show()


plot_history(history)  # 과적합

# 예측
x_test_s = scaler.transform(x_test)
y_test = y_test.values

y_pred = model.predict(x_test_s)

# 모양 맞춰주기
print(y_pred.shape)
print(y_test.shape)

y_pred = y_pred.reshape(-1)
print(mean_squared_error(y_test, y_pred))

# Confusion Matrix
# 0과 1로 변환 (이진분류)
y_pred = ((y_pred > 0.5).astype(int))


def plot_confusion_matrix(y_true, y_pred):
    cfm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(5, 5))
    sns.heatmap(cfm, annot=True, cbar=False, fmt="d")
    plt.xlabel("Predicted Class")
    plt.ylabel("True Class")
    plt.show()


plot_confusion_matrix(y_test, y_pred)
