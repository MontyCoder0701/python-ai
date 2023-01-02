# 선형회귀(경사하강법)

import numpy as np
import matplotlib.pyplot as plt

# 데이터 준비
x_train = np.array([1., 2., 3., 4., 5., 6.])
y_train = np.array([9., 12., 15., 18., 21., 24.])

plt.scatter(x_train, y_train)
# plt.show()

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

# Class 생성하기
y_pred = 3.0 * x_train + 6.0
print(y_pred)


class LinearRegressor:
    def fit(self, x, y, epochs=100, lr=0.01):
        self.w_ = 0.0
        self.b_ = 0.0
        self.n_ = len(x)
        self.lr_ = lr

        for i in range(epochs):
            # 가설
            h_hat = self.w_ * x_train + self.b_
            # cost(mse)
            cost = np.sum((y_train - h_hat)**2)/self.n_

            # 미분
            grad_w = np.sum(
                (self.w_ * x_train + self.b_ - y_train)*2 * x_train)/self.n_
            grad_b = np.sum((self.w_ * x_train + self.b_ - y_train)*2)/self.n_

            self.w_ = self.w_ - lr * grad_w
            self.b_ = self.b_ - lr * grad_b

            if i % 200 == 0:
                print(
                    f"Epoch: {i}, cost: {cost:10f}, w: {self.w_:10f}, b: {self.b_:10f}")

    def predict(self, x):
        return self.w_ * x + self.b_


lr = LinearRegressor()
lr.fit(x_train, y_train, epochs=5000)

x_test = np.array([10, 11])
y_pred = lr.predict(x_test)
print(y_pred)
