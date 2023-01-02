# 선형회귀 (Gradient Tape) - Tensorflow의 자동미분 기능

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# 데이터 준비
x_train = np.array([1., 2., 3., 4., 5., 6.])
y_train = np.array([9., 12., 15., 18., 21., 24.])

w = tf.Variable(0.0)
b = tf.Variable(0.0)

epochs = 5000
lr = 0.01

y_hat = w * x_train + b

# MSE
cost = tf.reduce_mean(tf.square(y_train-y_hat))

# 자동 미분기능
with tf.GradientTape() as tape:
    y_hat = w * x_train + b
    cost = tf.reduce_mean(tf.square(y_train-y_hat))

w_grad, b_grad = tape.gradient(cost, [w, b])

# w = w - lr * w_grad
w.assign_sub(lr * w_grad)
b.assign_sub(lr * b_grad)

print(w.numpy(), b.numpy())

for i in range(epochs):
    with tf.GradientTape() as tape:
        y_hat = w * x_train + b
        cost = tf.reduce_mean(tf.square(y_train-y_hat))

        w_grad, b_grad = tape.gradient(cost, [w, b])

        w.assign_sub(lr * w_grad)
        b.assign_sub(lr * b_grad)

        if i % 200 == 0:
            print(
                f"Epoch: {i}, cost: {cost.numpy():10f}, w: {w.numpy():10f}, b: {b.numpy():10f}")
