# CNN (Colored Image classification- Mask detection)
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
import PIL

# !pip install --upgrade --no-cache-dir gdown

# Data 준비
# !gdown https://drive.google.com/uc?id=12FIGFL_-WnKTTOo8AC_eh3O1VNzvDeQP
# !unzip MaskDatasets.zip


import os

data_root = "/content/MaskDatasets"
train_dir = os.path.join(data_root, "Train")
test_dir = os.path.join(data_root, "Test")

# 파일명 list 얻기
train_mask_fname = os.listdir(os.path.join(train_dir, "Mask"))
train_nomask_fname = os.listdir(os.path.join(train_dir, "NoMask"))
print("Mask:", len(train_mask_fname), "NoMask:", len(train_nomask_fname))

test_mask_fname = os.listdir(os.path.join(test_dir, "Mask"))
test_nomask_fname = os.listdir(os.path.join(test_dir, "NoMask"))
print("Mask:", len(test_mask_fname), "NoMask:", len(test_nomask_fname))

class_name = ["Mask", "NoMask"]

# 사진 얻기
x_train = []
y_train = []

for i in range(len(train_mask_fname)):
    f = os.path.join(train_dir, "Mask", train_mask_fname[i])
    image = PIL.Image.open(f)
    image = image.resize((224, 224))
    arr = np.array(image)
    x_train.append(arr)
    y_train.append(0)


for i in range(len(train_nomask_fname)):
    f = os.path.join(train_dir, "NoMask", train_nomask_fname[i])
    image = PIL.Image.open(f)
    image = image.resize((224, 224))
    arr = np.array(image)
    x_train.append(arr)
    y_train.append(1)


x_test = []
y_test = []

for i in range(len(test_mask_fname)):
    f = os.path.join(test_dir, "Mask", test_mask_fname[i])
    image = PIL.Image.open(f)
    image = image.resize((224, 224))
    arr = np.array(image)
    x_test.append(arr)
    y_test.append(0)


for i in range(len(test_nomask_fname)):
    f = os.path.join(test_dir, "NoMask", test_nomask_fname[i])
    image = PIL.Image.open(f)
    image = image.resize((224, 224))
    arr = np.array(image)
    x_test.append(arr)
    y_test.append(1)
# 시각화
samples = np.random.randint(300, size=9)

plt.figure(figsize=(8, 6))

for i, idx in enumerate(samples):
    plt.subplot(3, 3, i+1)
    plt.imshow(x_train[idx])
    plt.xticks([])
    plt.yticks([])
    plt.title(class_name[y_train[idx]])
plt.show()

# numpy로 변환
x_train = np.array(x_train)
y_train = np.array(y_train)
x_test = np.array(x_test)
y_test = np.array(y_test)

# 검증용 데이터
x_train, x_val, y_train, y_val = train_test_split(
    x_train, y_train, test_size=0.3)


# Min-max 정규화 (0~1 사이 값)
x_train = x_train.astype("float32")/255
x_val = x_val.astype("float32")/255
x_test = x_test.astype("float32")/255


# 모델 생성
print(x_train.shape)  # input shape 알기 위해

# Batch normalization, dropout 추가
model = keras.Sequential([
    layers.Conv2D(filters=16, kernel_size=3, input_shape=(
        224, 224, 3), activation="relu"),
    layers.MaxPool2D(2),
    # Overfitting 줄이기 위해 dropout
    # layers.Dropout(0.3),

    layers.Conv2D(filters=32, kernel_size=3, activation="relu"),
    layers.MaxPool2D(2),
    # layers.Dropout(0.3),

    layers.Conv2D(filters=64, kernel_size=3, activation="relu"),
    layers.MaxPool2D(2),
    # layers.Dropout(0.3),

    layers.Conv2D(filters=128, kernel_size=3, activation="relu"),
    layers.MaxPool2D(2),
    # layers.Dropout(0.3),

    # Dense 넣기 전에 flatten
    layers.Flatten(),
    layers.Dense(units=64, activation="relu"),
    # 이진 분류 모델 (sigmoid 출력층 - 0 혹은 1)/ 다중분류 (softmax)
    layers.Dense(units=1, activation="sigmoid"),
])

model.summary()

loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)

model.compile(
    optimizer="adam",
    loss=loss,
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

history = model.fit(x_train, y_train, epochs=epochs, callbacks=[ckpt_callback, es_callback],
                    batch_size=batch_size, validation_data=(x_val, y_val), verbose=1)

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
model.evaluate(x_train, y_train)
model.evaluate(x_test, y_test)

# 예측
y_pred = model.predict(x_test)

# 오답인 이미지 출력
samples = np.where((y_test == y_pred) == False)[0]
samples = np.random.choice(samples, 9)

# 시각화
plt.figure(figsize=(8, 6))

for i, idx in enumerate(samples):
    plt.subplot(3, 3, i+1)
    plt.imshow(x_test[idx])
    plt.xticks([])
    plt.yticks([])
    plt.title(class_name[y_test[idx]])
plt.show()
