# CNN (Colored Image classification- Cifar 10)
from keras import callbacks
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, mean_squared_error
import tensorflow as tf
from tensorflow import keras
from keras import layers
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
from keras.datasets import cifar10
from keras.applications import VGG16

# Prepare data
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
# print(x_train)
# print(y_train)
y_train = y_train.reshape(-1)

class_name = ["airplane", "automobile", "bird", "cat",
              "deer", "dog", " frog", "horse", "ship", "truck"]

# 시각화
samples = np.random.randint(len(x_train), size=9)

plt.figure(figsize=(8, 6))

for i, idx in enumerate(samples):
    plt.subplot(3, 3, i+1)
    plt.imshow(x_train[idx])
    plt.xticks([])
    plt.yticks([])
    plt.title(class_name[y_train[idx]])
plt.show()

# 검증용 데이터
x_train, x_val, y_train, y_val = train_test_split(
    x_train, y_train, test_size=0.3)

# Min-max 정규화 (0~1 사이 값)
x_train = x_train.astype("float32")/255
x_val = x_val.astype("float32")/255
x_test = x_test.astype("float32")/255

# Y값 One-hot 인코딩 (다중분류)
y_train_oh = to_categorical(y_train)
y_val_oh = to_categorical(y_val)
y_test_oh = to_categorical(y_test)

# 모델 생성
print(x_train.shape)  # input shape 알기 위해

base = VGG16(weights="imagenet", input_shape=(32, 32, 3), include_top=False)
base.trainable = False
base.summary()

model = keras.Sequential([
    base,
    layers.Flatten(),
    layers.Dense(64, activation="relu"),
    layers.Dense(10, activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["acc"]
)
model.summary()

# 학습
epochs = 30
batch_size = 64

history = model.fit(x_train, y_train_oh, epochs=epochs,
                    batch_size=batch_size, validation_data=(x_val, y_val_oh), verbose=1)

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

# 평가
model.evaluate(x_train, y_train_oh)
model.evaluate(x_test, y_test_oh)
