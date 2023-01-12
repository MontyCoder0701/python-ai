# IMDB 영화리뷰 이진분류

import re
from keras.utils import to_categorical, pad_sequences
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import keras
from keras import layers
from keras.datasets import imdb

(train_input, train_target), (test_input,
                              test_target) = imdb.load_data(num_words=1000)
print(imdb.get_word_index())

# index_to_word (sequence를 원래의 문장으로)
word_index = imdb.get_word_index()

index_to_word = dict(
    [(value, key) for (key, value) in word_index.items()]
)
print(index_to_word)

review = " ".join([index_to_word.get(i-3, "?") for i in train_input[0]])
print(review)

# 리뷰 길이 조사
review_len = np.array([len(x) for x in train_input])
review_len.min(), review_len.max()


plt.hist(review_len)
plt.xlabel("length")
plt.ylabel("frequency")
plt.show()

# Padding
train_seq = pad_sequences(train_input, maxlen=200, padding="pre")
print(train_seq.shape)

# 원핫인코딩
train_oh = to_categorical(train_seq)
print(train_oh.shape)

# 모델
# 모델
model = keras.Sequential([
    layers.Embedding(500, 16, input_length=200),  # One hot 필요 없음
    layers.LSTM(20, activation="tanh"),
    layers.Dense(1, activation="sigmoid")
])

model.compile(
    optimizer="rmsprop",
    loss="binary_crossentropy",
    metrics=["acc"]
)

# Callback (model checkpoint)
ckpt_path = "./temp/imdb.ckpt"
ckpt_callback = keras.callbacks.ModelCheckpoint(
    ckpt_path,
    monitor="val_loss",
    save_weights_only=True,
    save_best_only=True,
    verbose=1,
)

# 학습
epochs = 30
batch_size = 32

history = model.fit(train_oh, train_target, epochs=epochs, callbacks=[ckpt_callback],
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

# 저장된 최종 상태 가져오기
model.load_weights(ckpt_path)

# 평가
model.evaluate(train_seq, train_target)

# Testing

review_sentence = "This was the best movie I ever saw! I want to see it again. it is quite amazing, and I definitely recommend it to whoever is bored."

# 문장 처리리
word_to_index = imdb.get_word_index()
index_to_word = {}

for key, value in word_to_index.items():
    index_to_word[value+3] = key

index_to_word[0] = "<PAD>"
index_to_word[1] = "<SOS>"
index_to_word[2] = "<OOV>"

review_sentence = re.sub("[^0-9a-zA-Z ]", "", review_sentence)  # non letter 처리
review_sentence = review_sentence.lower()
print(review_sentence)

encoded = []

for word in review_sentence:
    try:
        if word_to_index[word] <= 500:
            encoded.append(word_to_index[word]+3)
        else:
            encoded.append(2)
    except KeyError:
        encoded.append(2)

print(encoded)

# Padding
pad_new = pad_sequences([encoded], maxlen=200)

# 결과
y_pred = model.predict(pad_new)
print(y_pred)
