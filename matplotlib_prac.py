import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

print(matplotlib.__version__)

x = [1, 3, 5]
y = [2, 8, 11]

# plt.plot(x, y)
print(plt.rcParams.get("figure.figsize"))
# plt.title("Graph")
# plt.xlabel("This is x", loc="right")
# plt.ylabel("This is y")

# Figure, Axes
# fig = plt.figure()
# ax1 = fig.add_subplot(2, 2, 1)
# ax2 = fig.add_subplot(2, 2, 2)
# ax3 = fig.add_subplot(2, 2, 3)
# ax4 = fig.add_subplot(2, 2, 4)

# ax1.plot(x, y)
# ax1.set_title("Graph")

# ax4.plot(x, y, marker="*", label="Eng")

# z = [3, 4, 5]

# plt.plot(x, z, label="Math")
# plt.legend()

# 막대그래프 barplot
x = ["A", "B", "C", "D", "E"]
y = [3, 2, 4, 5, 6]
# plt.bar(x, y, color="pink")

# 히스토그램
my_data = np.random.randn(500)
# plt.hist(my_data, bins=20)

# Scatter plot (Iris)
df = pd.read_csv("iris.csv", index_col=0)
print(df.shape)
print(df.head())
# plt.scatter(x=df["SepalLengthCm"], y=df["SepalWidthCm"])

fig = plt.figure()
plt.title("Iris Dataset")

ax1 = fig.add_subplot(1, 2, 1)
ax2 = fig.add_subplot(1, 2, 2)

ax1.set_xlabel("PetalLengthCm")
ax1.set_ylabel("PetalWidthCm")

print(df["Species"].unique())
print(df["Species"].value_counts())

ax1.scatter(x=df.loc[df["Species"] == "Iris-setosa",
            "PetalLengthCm"], y=df.loc[df["Species"] == "Iris-setosa",
            "PetalWidthCm"], color="red")

ax1.scatter(x=df.loc[df["Species"] == "Iris-versicolor",
            "PetalLengthCm"], y=df.loc[df["Species"] == "Iris-versicolor",
            "PetalWidthCm"], color="green")

ax1.scatter(x=df.loc[df["Species"] == "Iris-virginica",
            "PetalLengthCm"], y=df.loc[df["Species"] == "Iris-virginica",
            "PetalWidthCm"], color="blue")

ax2.set_xlabel("SepalLengthCm")
ax2.set_ylabel("SepalWidthCm")

ax2.scatter(x=df.loc[df["Species"] == "Iris-setosa",
            "SepalLengthCm"], y=df.loc[df["Species"] == "Iris-setosa",
            "SepalWidthCm"], color="red")

ax2.scatter(x=df.loc[df["Species"] == "Iris-versicolor",
            "SepalLengthCm"], y=df.loc[df["Species"] == "Iris-versicolor",
            "SepalWidthCm"], color="green")

ax2.scatter(x=df.loc[df["Species"] == "Iris-virginica",
            "SepalLengthCm"], y=df.loc[df["Species"] == "Iris-virginica",
            "SepalWidthCm"], color="blue")

my_data = [[1, 2, 3, 4], [10, 11, 12, 13], [41, 42, 43, 44], [55, 56, 57, 58]]
fig, ax = plt.subplots(2, 2)

ax = ax.reshape(-1)  # flatten

# ax[0, 0]. plot(my_data[0])
# ax[0, 1]. plot(my_data[1])
# ax[1, 0]. plot(my_data[2])
# ax[1, 1]. plot(my_data[3])

for i in range(4):
    ax[i].plot(my_data[i])

# for x, z in zip(ax, my_data):
#     x.plot(z)

# 이미지 표시
fig, ax = plt.subplots(2, 2)
plt.title("Cute animals")

img1 = Image.open("cat.jpg")
img2 = Image.open("dog1.jpg")
img3 = Image.open("dog2.jpg")

ax[0, 0].imshow(img1)
ax[0, 1].imshow(img2)
ax[1, 0].imshow(img3)

plt.figure(figsize=(5, 5))
# plt.imshow(img)
plt.xticks([])
plt.yticks([])

# arr = np.array(img)
# print(arr)
# print(arr.shape)

plt.show()
