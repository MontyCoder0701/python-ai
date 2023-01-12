from keras.utils import to_categorical
from keras.utils import pad_sequences
from keras.preprocessing.text import Tokenizer

# Embedding
corpus = {
    "I love my dog",
    "I love my cat",
    "You love my dog",
    "Do you think my cat is amazing",
}

tokenizer = Tokenizer(num_words=100, oov_token="<OOV>")
tokenizer.fit_on_texts(corpus)
tokenizer.word_index
tokenizer.word_counts

# 영어 문장을 숫자로 변환
sequences = tokenizer.texts_to_sequences(corpus)
print(sequences)

padded = pad_sequences(sequences, maxlen=6, padding="pre")
print(padded)

# 원핫인코딩
padded_seq = to_categorical(padded)
print(padded_seq)
