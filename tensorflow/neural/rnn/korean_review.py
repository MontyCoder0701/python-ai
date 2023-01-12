# 한국어 Naver 영화리뷰 이진분류

from keras.preprocessing.text import Tokenizer
from konlpy.tag import Okt
from tqdm import tqdm
import re
from keras.utils import to_categorical, pad_sequences
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import keras
from keras import layers
from keras.datasets import imdb

# apt-get update
# apt-get install g++ openjdk-8-jdk python-dev python3-dev
# pip install JPype1
# pip install konlpy
# JAVA_HOME="/usr/lib/jvm/java-8-openjdk-amd64"

# 데이터 준비
# !git clone https://github.com/e9t/nsmc

train_data = pd.read_csv(
    "/content/nsmc/ratings_train.txt", sep="\t")  # seperator 써야함

# 중복치, 결측치 처리
train_data.isna().sum(axis=0)
train_data["document"].duplicated().sum()
train_data.drop_duplicates(subset=["document"], inplace=True)

# 특수기호 처리
train_data["document"] = train_data["document"].str.replace("[^가-힣 ]", "")
train_data["document"] = train_data["document"].str.replace("^ +", "")
train_data["document"] = train_data["document"].replace("", np.nan)
train_data = train_data.dropna()
print(train_data)

# 토큰화

stopwords = {"의", "가", "이", "은", "들", "자", "는", "영화"}
x_train = []

for sentence in tqdm(train_data["document"]):
    okt = Okt()
    temp_x = okt.morphs(sentence, stem=True)
    temp_x = [word for word in temp_x if not word in stopwords]
    x_train.append(temp_x)

# 정수 인코딩

tokenizer = Tokenizer()
tokenizer.fit_on_texts(x_train)

x_train = tokenizer.texts_to_sequences(x_train)
y_train = train_data["label"].values

print(x_train, y_train)


# 불필요한 데이터 제거
drop_train = []

for index, sentence in enumerate(x_train):
    if len(sentence) < 2:
        drop_train.append(index)

x_train = np.delete(x_train, drop_train, axis=0)

# 패딩
review_len = [len(x) for x in x_train]
plt.figure(figsize=(10, 7))
plt.hist(review_len, bins=50)
plt.show()

x_train = pad_sequences(x_train, maxlen=30)
print(x_train)


# 모델
model = keras.Sequential([
    layers.Embedding(43080, 16, input_length=30),
    layers.LSTM(128),
    layers.Dense(1, activation="sigmoid")
])

model.compile(
    optimizer="rmsprop",
    loss="binary_crossentropy",
    metrics=["acc"]
)


# 학습
epochs = 30
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
    plt.ylabel('Accuracy')
    plt.plot(hist['epoch'], hist['acc'], label='Train Accuracy')
    plt.plot(hist['epoch'], hist['val_acc'], label='Val Accuracy')
    plt.legend()

    plt.show()


plot_history(history)
