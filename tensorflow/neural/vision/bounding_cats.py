# CNN (Colored Image classification- Bounding box cats)
from keras.applications import MobileNetV2
import cv2
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


# !gdown https://drive.google.com/uc?id=1-RBvPOYycsSpS7rVP0Pqwcbh18lZYDeb
# !unzip /content/BBRegression.zip

# Data 준비
import glob
import xml.etree.ElementTree as ET
import os

# 함수 가져오기


def xml_to_csv(path):
    xml_list = []
    for xml_file in glob.glob(path + '/*.xml'):

        tree = ET.parse(xml_file)
        root = tree.getroot()
        for member in root.findall('object'):
            bbx = member.find('bndbox')
            xmin = int(bbx.find('xmin').text)
            ymin = int(bbx.find('ymin').text)
            xmax = int(bbx.find('xmax').text)
            ymax = int(bbx.find('ymax').text)
            label = member.find('name').text

            value = (root.find('filename').text,
                     int(root.find('size')[0].text),
                     int(root.find('size')[1].text),
                     label,
                     xmin,
                     ymin,
                     xmax,
                     ymax
                     )
            xml_list.append(value)
    column_name = ['filename', 'width', 'height',
                   'class', 'xmin', 'ymin', 'xmax', 'ymax']
    xml_df = pd.DataFrame(xml_list, columns=column_name)
    return xml_df


image_path = "/content/BBRegression"
file_name = "label_cats.csv"
csv_path = os.path.join(image_path, "train")
xml_df = xml_to_csv(csv_path)
xml_df.to_csv(file_name)

images = xml_df.iloc[:, 0].values
points = xml_df.iloc[:, 4:].values

# 시각화

dataset_images = []
dataset_bbs = []

for file, points in zip(images, points):
    f = os.path.join(image_path, "train", file)
    image = PIL.Image.open(f)
    arr = np.array(image)
    dataset_images.append(arr)
    dataset_bbs.append(points)

dataset_images = np.array(dataset_images)
dataset_bbs = np.array(dataset_bbs)

print(dataset_images.shape)
print(dataset_bbs.shape)


samples = np.random.randint(9, size=4)
plt.figure(figsize=(8, 8))
for i, idx in enumerate(samples):
    points = dataset_bbs[idx].reshape(2, 2)
    img = cv2.rectangle(dataset_images[idx].copy(),
                        tuple(points[0]),
                        tuple(points[1]),
                        color=(255, 0, 0),
                        thickness=2,
                        )
    plt.subplot(2, 2, i+1)
    plt.imshow(img)
plt.show()

np.savez("cat_bbs.npz",
         image=dataset_images,
         bbs=dataset_bbs)

dataset = np.load("cat_bbs.npz")

x = dataset["image"]
y = dataset["bbs"]

x.shape, y.shape

# 검증데이터

x_train, x_val, y_train, y_val = train_test_split(
    x, y, test_size=0.3)

# 정규화
x_train = x_train.astype("float32")/225
x_val = x_val.astype("float32")/225

y_train = y_train.astype("float32")
y_val = y_val.astype("float32")


# 모델 생성
print(x_train.shape)  # input shape 알기 위해

base = MobileNetV2(input_shape=(224, 224, 3),
                   weights="imagenet", include_top=False)
base.trainable = False
base.summary()

model = keras.Sequential([
    base,
    # base 후 flatten
    layers.Flatten(),
    layers.Dense(128, activation="relu"),
    layers.Dense(64, activation="relu"),
    layers.Dense(4)
])

# 회귀의 경우 mse
model.compile(
    optimizer="adam",
    loss="mse",
    metrics=["mse"]
)
model.summary()

# Callback (model checkpoint)
ckpt_path = "cats_bbs.ckpt"
ckpt_callback = callbacks.ModelCheckpoint(
    ckpt_path,
    save_weights_only=True,
    save_best_only=True,
    monitor="val_loss",
    verbose=1,
)

# 학습
epochs = 30
batch_size = 16

history = model.fit(x_train, y_train, epochs=epochs, callbacks=[ckpt_callback],
                    batch_size=batch_size, validation_data=(x_val, y_val), verbose=1)


# matplotlib

def plot_history(history):
    hist = pd.DataFraitjy6u7me(history.history)
    hist["epoch"] = history.epoch
    plt.plot(hist["epoch"], hist["mse"], label="Train MSE")
    plt.plot(hist["epoch"], hist["val_mse"], label="Val MSE")
    plt.legend()
    plt.show()


plot_history(history)

# 모델 저장
# keras
# model.save("cats_bbs_regression.h5")
# my_model = keras.models.load_model("cats_bbs_regression.h5")

# tensorflow
# model.save("my_cats_model")
# my_model = keras.models.load_model("my_cats_model")
# my_model.summary()

# 테스트 이미지 로딩
fnames = glob.glob("/content/BBRegression/test" + "/*.jpg")

# 사진 얻기
x_test = []

for f in fnames:
    image = PIL.Image.open(f)
    arr = np.array(image)
    x_test.append(arr)
x_test = np.array(x_test)

x_test = x_test.astype("float32")/255
y_pred = model.predict(x_test).astype(int)


# 곃과 시각화
samples = np.random.randint(9, size=4)
plt.figure(figsize=(8, 8))
for i, idx in enumerate(samples):
    points = y_pred[idx].reshape(2, 2)
    img = cv2.rectangle(x_test[idx].copy(),
                        tuple(points[0]),
                        tuple(points[1]),
                        color=(255, 0, 0),
                        thickness=2,
                        )
    plt.subplot(2, 2, i+1)
    plt.imshow(img)
plt.show()
