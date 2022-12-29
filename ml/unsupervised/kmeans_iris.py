# K-means, iris
from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Data (make own sample)
iris = pd.read_csv('basics/iris.csv')
df = iris.drop(['Id'], axis=1).copy()

x = df.drop(['Species'], axis=1)
y = df['Species']

# 학습
km = KMeans(n_clusters=3, random_state=2022)
km.fit(x)

# 결과 보기
y_pred = km.predict(x)
print(y_pred)

df = pd.DataFrame(x, columns=["SepalLengthCm", "SepalWidthCm"])
df["y_pred"] = y_pred
print(df.head())

# Visualize
sns.scatterplot(data=df, x="SepalLengthCm", y="SepalWidthCm", hue="y_pred")

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
