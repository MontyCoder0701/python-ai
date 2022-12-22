# KNN Citrus Dataset (Binary Classification)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.utils import shuffle
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("ml/citrus.csv")

# See data info
print(df.isna().sum())
print(df.duplicated().sum())
print(df.describe().T)

# sns.pairplot(df)

df['name'] = df['name'].map({
    'orange': 0,
    'grapefruit': 1,
})

x = df.drop(['name'], axis=1)
y = df['name']

# Shuffle, split dataset
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=2022)

print(x_train.shape, x_test.shape)
print(y_train.shape, y_test.shape)

# x, y = shuffle(x, y, random_state=2022)
# num = int(len(y)*0.8)

# x_train = x.iloc[:num, :]
# x_test = x.iloc[num:, :]
# y_train = y.iloc[:num]
# y_test = y.iloc[num:]

# 정규화
# for col in x_train.columns:
#     mu = x_train[col].mean()
#     std = x_train[col].std()
#     x_train[col] = (x_train[col] - mu)/std
#     x_test[col] = (x_test[col] - mu)/std

scaler = StandardScaler()
scaler.fit(x_train)
x_train_s = scaler.transform(x_train)
x_test_s = scaler.transform(x_test)

y_train = y_train.values
y_test = y_test.values

scores = []

for i in range(3, 30):
    clf = KNeighborsClassifier(n_neighbors=i)
    clf.fit(x_train_s, y_train)
    s = clf.score(x_train_s, y_train)
    scores.append(s)

plt.plot(scores)
plt.show()
print(scores)

# Best model
clf = KNeighborsClassifier(n_neighbors=10)
clf.fit(x_train_s, y_train)
y_pred = clf.predict(x_test_s)


def print_score(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    pre = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)

    print("accuracy: ", acc)
    print("precision: ", pre)
    print("recall: ", rec)


print_score(y_test, y_pred)

cfm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(5, 5))
sns.heatmap(cfm, annot=True, cbar=False, fmt="d")
plt.xlabel("Predicted Class")
plt.ylabel("True Class")
plt.show()
