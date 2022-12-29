import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV

# 주성분 분석 (PCA with Iris dataset)
df = pd.read_csv("ml/svm/train.csv")

# y값 인코딩
df['Activity'] = df['Activity'].map({
    'STANDING': 0,
    'SITTING': 1,
    'LAYING': 2,
    'WALKING': 3,
    'WALKING_DOWNSTAIRS': 4,
    'WALKING_UPSTAIRS': 5,
})

# x, y 분리
x_train = df.drop(['Activity'], axis=1)
y_train = df['Activity']

# 표준정규화
scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
y_train = y_train.values

# 차원 축소
pca = PCA(n_components=150)
x_train = pca.fit_transform(x_train)
df["y_train"] = y_train

# 분류 작업 Random Forest
df_x = df
df_y = df['y_train']

clf = RandomForestClassifier()
clf.fit(df_x, df_y)
print(clf.score(df_x, df_y))
