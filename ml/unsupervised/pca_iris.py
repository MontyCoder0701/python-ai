import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier, plot_tree

# 주성분 분석 (PCA with Iris dataset)
iris = datasets.load_iris()

# Df로 변환
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df["target"] = iris.target

# x,y 분리
x = df.drop(['target'], axis=1)
y = df['target']

# 학습셋 분리
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, stratify=y, random_state=2022)

# 표준화
scaler = StandardScaler()
scaler.fit_transform(x_train)
y_train = y_train.values

# 차원 축소
pca = PCA(n_components=2)
x_train = pca.fit_transform(x_train)
print(x_train)

# Visualize
df = pd.DataFrame(x_train, columns=["x_1", "x_2"])
df["y_train"] = y_train

sns.scatterplot(data=df, x="x_1", y="x_2", hue="y_train")
plt.show()

# 분류 작업 Decision Tree
df_x = df[["x_1", "x_2"]]
df_y = df['y_train']

clf = DecisionTreeClassifier(max_depth=3)
clf.fit(df_x, df_y)
print(clf.score(df_x, df_y))
