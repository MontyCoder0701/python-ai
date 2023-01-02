# 선형회귀(경사하강법)

import numpy as np
import matplotlib.pyplot as plt

# 데이터 준비
x_train = np.array([1., 2., 3., 4., 5., 6.])
y_train = np.array([9., 12., 15., 18., 21., 24.])

plt.scatter(x_train, y_train)
plt.show()

# 학습
# 경사하강법 (gradient descent)

w = 0.0
b = 0.0
n = len(x_train)

epochs = 5000
lr = 0.01

for i in range(epochs):
    # 가설
    h_hat = w * x_train + b
    # cost(mse)
    cost = np.sum((y_train - h_hat)**2)/n

    # 미분
    grad_w = np.sum((w * x_train + b - y_train)*2 * x_train)/n
    grad_b = np.sum((w * x_train + b - y_train)*2)/n

    w = w - lr * grad_w
    b = b - lr * grad_b

    if i % 200 == 0:
        print(f"Epoch: {i}, cost: {cost:10f}, w: {w:10f}, b: {b:10f}")

# 3x+6으로 결과 나옴 (cost=0, w, b)
