# Image augmentation

from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
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

import os

data_root = "/content/cats_and_dogs_filtered"
train_dir = os.path.join(data_root, "train")
val_dir = os.path.join(data_root, "validation")


train_gen = ImageDataGenerator(
    rescale=1./255.,
    rotation_range=90,
    width_shift_range=0.4,
    height_shift_range=0.4,
    vertical_flip=True,
    horizontal_flip=True,
    validation_split=0.2,
)

test_gen = ImageDataGenerator(rescale=1./255.,)

batch_size = 32
image_size = (224, 224)

train_iter = train_gen.flow_from_directory(
    train_dir,
    batch_size=batch_size,
    target_size=image_size,
    class_mode="binary",
    subset="training",
)

val_iter = train_gen.flow_from_directory(
    train_dir,
    batch_size=batch_size,
    target_size=image_size,
    class_mode="binary",
    subset="validation",
)

test_iter = test_gen.flow_from_directory(
    val_dir,
    batch_size=batch_size,
    target_size=image_size,
    class_mode="binary",
)

# Batch size
images, labels = train_iter.next()
print(len(images), len(labels))

plt.figure(figsize=(12, 8))
for i in range(15):
    plt.subplot(3, 5, i+1)
    plt.axis("off")
    plt.imshow(images[i])
    plt.title(labels[i])
plt.show()

# 모델 생성
model = keras.Sequential([
    layers.Conv2D(filters=32, kernel_size=3, padding="same",
                  activation="relu", input_shape=(224, 224, 3)),
    layers.MaxPool2D(pool_size=(2, 2)),
    layers.Conv2D(filters=64, padding="same", kernel_size=3,
                  activation="relu"),
    layers.MaxPool2D(pool_size=(2, 2)),
    layers.Conv2D(filters=128, padding="same", kernel_size=3,
                  activation="relu"),
    layers.MaxPool2D(pool_size=(2, 2)),
    # Dense 넣기 전에 flatten
    layers.Flatten(),

    layers.Dense(units=256, activation="relu"),
    # 이진 분류 모델 (sigmoid 출력층 - 0 혹은 1)/ 다중분류 (softmax)
    layers.Dense(units=1, activation="sigmoid"),
])

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["acc"]
)

model.summary()

# 학습
epochs = 3

history = model.fit(train_iter, epochs=epochs,
                    validation_data=val_iter, verbose=1)

# 평가
model.evaluate(train_iter)
model.evaluate(test_iter)
