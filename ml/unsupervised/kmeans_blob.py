# K-means, sample data
from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Data (make own sample)
from sklearn.datasets import make_blobs
x, y = make_blobs(n_samples=150, centers=3, n_features=2,
                  cluster_std=0.5, random_state=0)

# Visualize
plt.scatter(x[:, 0], x[:, 1])
plt.show()

# 학습
km = KMeans(n_clusters=3, random_state=2022)
km.fit(x)

# 결과 보기
y_pred = km.predict(x)
print(y_pred)

df = pd.DataFrame(x, columns=["x_1", "x_2"])
df["y_pred"] = y_pred
print(df.head())

# Visualize
sns.scatterplot(data=df, x="x_1", y="x_2", hue="y_pred")

centroid = km.cluster_centers_
plt.scatter(centroid[:, 0], centroid[:, 1], s=150, marker="*", c="red")

plt.show()

# Elbow method
inertia = []

for k in range(2, 11):
    km = KMeans(n_init="auto", n_clusters=k, random_state=2022)
    km.fit(x)
    inert = km.inertia_
    inertia.append(inert)

print(inertia)
plt.plot(range(2, 11), inertia, marker="o")
plt.show()
