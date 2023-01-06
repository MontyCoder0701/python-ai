# CNN (BW Image classification- Fashion Dataset)
from keras.models import Model
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
from keras.datasets import fashion_mnist
from keras.utils import to_categorical

# Data 가져오기
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

class_name = ["T-shirt/top", "Trouser", "Pullover", "Dress",
              "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

# 시각화
samples = np.random.randint(60000, size=9)

plt.figure(figsize=(8, 6))

for i, idx in enumerate(samples):
    plt.subplot(3, 3, i+1)
    plt.imshow(x_train[idx], cmap="gray")
    plt.xticks([])
    plt.yticks([])
    plt.title(class_name[y_train[idx]])
# plt.show()

# 검증용 데이터
x_train, x_val, y_train, y_val = train_test_split(
    x_train, y_train, test_size=0.3)

# Min-max 정규화 (0~1 사이 값)
x_train = x_train.astype("float32")/255
x_val = x_val/255
x_text = x_test/255

# Y값 One-hot 인코딩 (다중분류)
y_train_oh = to_categorical(y_train)
y_val_oh = to_categorical(y_val)
y_test_oh = to_categorical(y_test)

# 모델 생성
# x를 1차원으로 변경해야 모델에 넣을 수 있음
x_train = x_train.reshape(-1, 28*28)
x_val = x_val.reshape(-1, 28*28)
x_test = x_test.reshape(-1, 28*28)

print(x_train.shape)

# Add convolution layer
# CNN에 넣기 위해 x 모양 바꾸기
x_train = x_train.reshape(-1, 28, 28, 1)
x_val = x_val.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)

# 함수형 API (Sequential로 만들 수 없는 모델 만들 경우)


def build_model():
    inputs = layers.Input(shape=(28, 28, 1))
    x1 = layers.Conv2D(filters=32, kernel_size=3,
                       activation="relu")(inputs)
    x2 = layers.MaxPool2D(pool_size=(2, 2))(x1)
    x3 = layers.Conv2D(filters=32, kernel_size=3,
                       activation="relu")(x2)
    x4 = layers.MaxPool2D(pool_size=(2, 2))(x3)
    x5 = layers.Flatten()(x4)
    x6 = layers.Dense(units=32, activation="relu")(x5)
    outputs = layers.Dense(units=10, activation="softmax")(x6)

    model = Model(inputs=inputs, outputs=outputs)

    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["acc"]
    )
    return model


model = build_model()

# 함수형 예제
input_x = layers.Input(shape=(16,))
x1 = layers.Dense(8, activation="relu")(input_x)
x2 = layers.Dense(16, activation="relu")(x1)
x3 = Model(inputs=input_x, outputs=x2)

input_y = layers.Input(shape=(16,))
x4 = layers.Dense(8, activation="relu")(input_y)
x5 = layers.Dense(16, activation="relu")(x4)
x6 = Model(inputs=input_y, outputs=x5)

# 합치기
combined = layers.concatenate([x3.output, x6.output])
z1 = layers.Dense(2, activation="relu")(combined)
z2 = layers.Dense(1)(z1)

model = Model(inputs=[x3.input, x6.input], outputs=z2)

# Sub classing (모델을 class로 만드는 것)


# class MnistModel(Model):
#     def __init__(self):
#         super(MnistModel, self).__init__()
#         self.conv1 = layers.Conv2D(64, 3, activation="relu")
#         self.conv2 = layers.Conv2D(32, 3, activation="relu")
#         self.maxpool = layers.MaxPool2D(2)
#         self.flat = layers.Flatten()
#         self.dense1 = layers.Dense(32, activation="relu")
#         self.dense2 = layers.Dense(10, activation="softmax")

#     def call(self, inputs):
#         x = self.conv1(inputs)
#         x = self.conv2(x)
#         x = self.maxpool(x)
#         x = self.flat(x)
#         x = self.dense1(x)
#         output = self.dense2(x)
#         return output


# model = MnistModel()


# 일반 sequential
model = keras.Sequential([
    layers.Conv2D(filters=32, kernel_size=3,
                  activation="relu", input_shape=(28, 28, 1)),
    layers.MaxPool2D(pool_size=(2, 2)),
    layers.Conv2D(filters=32, kernel_size=3,
                  activation="relu"),
    layers.MaxPool2D(pool_size=(2, 2)),
    # Dense 넣기 전에 flatten
    layers.Flatten(),

    layers.Dense(units=32, activation="relu"),
    # 이진 분류 모델 (sigmoid 출력층 - 0 혹은 1)/ 다중분류 (softmax)
    layers.Dense(units=10, activation="softmax"),
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

history = model.fit(x_train, y_train_oh, epochs=epochs,
                    batch_size=batch_size, validation_data=(x_val, y_val_oh), verbose=1)

# 평가
model.evaluate(x_train, y_train_oh)
model.evaluate(x_test, y_test_oh)

# 예측
y_pred = model.predict(x_test)

# One-hot 전으로 되돌리기 (정답값의 형태로)
y_pred = np.argmax(y_pred, axis=1)

# Confusion Matrix


def plot_confusion_matrix(y_true, y_pred):
    cfm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(5, 5))
    sns.heatmap(cfm, annot=True, cbar=False, fmt="d")
    plt.xlabel("Predicted Class")
    plt.ylabel("True Class")
    plt.show()


plot_confusion_matrix(y_test, y_pred)

# 오답인 이미지 출력
samples = np.where((y_test == y_pred) == False)[0]
samples = np.random.choice(samples, 9)

# 시각화
plt.figure(figsize=(8, 6))

# 다시 x를 이차원으로 reshape

for i, idx in enumerate(samples):
    plt.subplot(3, 3, i+1)
    plt.imshow(x_test[idx].reshape(28, 28), cmap="gray")
    plt.xticks([])
    plt.yticks([])
    plt.title(class_name[y_test[idx]])
plt.show()
