# Multi classification (Penguin dataset)
from keras.utils import to_categorical
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
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv('ml/knn/penguins.csv')

df.dropna(subset=["bill_length_mm"], inplace=True)
df["sex"].fillna("NONE", inplace=True)

island_encoder = LabelEncoder()
df["island"] = island_encoder.fit_transform(df["island"])

sex_encoder = LabelEncoder()
df["sex"] = sex_encoder.fit_transform(df["sex"])

species_encoder = LabelEncoder()
df["species"] = species_encoder.fit_transform(df["species"])

# X, Y 분리
x = df.drop(['species'], axis=1)
y = df['species']

# 섞고 test, train 나누기
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=2022)

# 표준화
scaler = StandardScaler()
x_train_s = scaler.fit_transform(x_train)
y_train = y_train.values

# 다중분류인 경우, y_train 값 변경 (one-hot encoding)
y_train_oh = to_categorical(y_train)
print(y_train_oh)

# 모델
model = keras.Sequential([
    layers.Dense(units=64, activation="relu", input_shape=(6,)),
    layers.Dense(units=32, activation="relu"),
    # 이진 분류 모델 (sigmoid 출력층 - 0 혹은 1)/ 다중분류 (softmax)
    layers.Dense(units=3, activation="softmax"),  # 다중분류 개수 만큼 최종 출력
])

model.summary()

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["acc"]
)

# 학습
epochs = 30
batch_size = 32

history = model.fit(x_train_s, y_train_oh, epochs=epochs,
                    batch_size=batch_size, validation_split=0.2, verbose=1)

# 평가
model.evaluate(x_train_s, y_train_oh)

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
    plt.plot(hist['epoch'], hist['acc'], label='Train Accuracy')
    plt.plot(hist['epoch'], hist['val_acc'], label='Val Accuracy')
    plt.legend()

    plt.show()


plot_history(history)

# 예측
x_test_s = scaler.transform(x_test)
y_test = y_test.values
y_test_oh = to_categorical(y_test)

y_pred = model.predict(x_test_s)

print(mean_squared_error(y_test_oh, y_pred))

# Confusion Matrix
# One-hot 전으로 되돌리기 (정답값의 형태로)
y_pred = np.argmax(y_pred, axis=1)


def plot_confusion_matrix(y_true, y_pred):
    cfm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(5, 5))
    sns.heatmap(cfm, annot=True, cbar=False, fmt="d")
    plt.xlabel("Predicted Class")
    plt.ylabel("True Class")
    plt.show()


plot_confusion_matrix(y_test, y_pred)
