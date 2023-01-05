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

# Batch normalization, dropout 추가
model = keras.Sequential([
    layers.Conv2D(filters=16, kernel_size=3, input_shape=(32, 32, 3)),
    layers.BatchNormalization(),
    layers.ReLU(),
    layers.MaxPool2D(2),
    # Overfitting 줄이기 위해 dropout
    # layers.Dropout(0.3),

    layers.Conv2D(filters=64, kernel_size=3),
    layers.BatchNormalization(),
    layers.ReLU(),
    layers.MaxPool2D(2),
    # layers.Dropout(0.3),

    layers.Conv2D(filters=128, kernel_size=3),
    layers.BatchNormalization(),
    layers.ReLU(),
    layers.MaxPool2D(2),
    # layers.Dropout(0.3),

    # Dense 넣기 전에 flatten
    layers.Flatten(),
    layers.Dropout(0.5),

    layers.Dense(units=64, activation="relu"),
    # 이진 분류 모델 (sigmoid 출력층 - 0 혹은 1)/ 다중분류 (softmax)
    layers.Dense(units=10, activation="softmax"),
])

model.summary()

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["acc"]
)

# Callback (model checkpoint)
ckpt_path = "cifar10.ckpt"
ckpt_callback = callbacks.ModelCheckpoint(
    ckpt_path,
    monitor="val_loss",
    save_weights_only=True,
    save_best_only=True,
    verbose=1,
)

# Early stopping
es_callback = callbacks.EarlyStopping(
    monitor="val_loss",
    patience=5,
    verbose=1,
)


# 학습
epochs = 30
batch_size = 32

history = model.fit(x_train, y_train_oh, epochs=epochs, callbacks=[ckpt_callback, es_callback],
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

# 저장된 최종 상태 가져오기
model.load_weights(ckpt_path)

# 평가
model.evaluate(x_train, y_train_oh)
model.evaluate(x_test, y_test_oh)
