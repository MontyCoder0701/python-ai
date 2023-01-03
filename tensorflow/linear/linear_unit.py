import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from keras import layers


model = keras.Sequential([
    # 완전연결층(Fully connected layer), first layer
    layers.Dense(units=1, input_shape=(1,)),
])

model.summary()

# adam = tf.keras.optimizers.Adam(learning_rate=0.01)
sgd = tf.keras.optimizers.SGD(learning_rate=0.01)

# Compile
model.compile(
    optimizer=sgd,
    loss="mse",
    metrics=["mse", "mae"]
)

# 학습
x_train = np.array([1., 2., 3., 4., 5., 6.])
y_train = np.array([9., 12., 15., 18., 21., 24.])

model.fit(x_train, y_train, epochs=5000, batch_size=len(x_train), verbose=1)

# 평가
model.evaluate(x_train, y_train)

# 예측
x_test = np.array([10.])
y_pred = model.predict(x_test)
print(y_pred)
