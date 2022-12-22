# KNN Iris Dataset (Multiclass Classification)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.utils import shuffle
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.metrics import confusion_matrix


def get_iris(mode=None):
    iris = pd.read_csv('basics/iris.csv')
    df = iris.drop(['Id'], axis=1).copy()

    # Column 명칭
    df.columns = ['sepal_length', 'sepal_width',
                  'petal_length', 'petal_width', 'species']

    # 이진분류
    if (mode == 'bin'):
        df = df.loc[df['species'] != 'Iris-virginica']

    df['species'] = df['species'].map({
        'Iris-setosa': 0,
        'Iris-versicolor': 1,
        'Iris-virginica': 2
    })

    # x, y 지정
    x = df.drop(['species'], axis=1)
    y = df['species']

    # 섞고 test, train 나누기
    x, y = shuffle(x, y, random_state=2022)
    num = int(len(y)*0.8)

    x_train = x.iloc[:num, :]
    x_test = x.iloc[num:, :]
    y_train = y.iloc[:num]
    y_test = y.iloc[num:]

    # 정규화
    for col in x_train.columns:
        mu = x_train[col].mean()
        std = x_train[col].std()
        x_train[col] = (x_train[col] - mu)/std
        x_test[col] = (x_test[col] - mu)/std

    return x_train, x_test, y_train, y_test


x_train, x_test, y_train, y_test = get_iris()
print(x_train.shape, x_test.shape)
print(y_train.shape, y_test.shape)

x_train = x_train.values
y_train = y_train.values

x_test = x_test.values
y_test = y_test.values

scores = []

for i in range(3, 30):
    clf = KNeighborsClassifier(n_neighbors=i)
    clf.fit(x_train, y_train)
    s = clf.score(x_train, y_train)
    scores.append(s)

plt.plot(scores)
plt.show()

clf = KNeighborsClassifier(n_neighbors=12)
clf.fit(x_train, y_train)
y_pred = clf.predict(x_test)


def print_score(y_true, y_pred, average="binary"):
    acc = accuracy_score(y_true, y_pred)
    pre = precision_score(y_true, y_pred, average=average)
    rec = recall_score(y_true, y_pred, average=average)

    print("accuracy: ", acc)
    print("precision: ", pre)
    print("recall: ", rec)


print_score(y_test, y_pred, average="macro")

cfm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(5, 5))
sns.heatmap(cfm, annot=True, cbar=False, fmt="d")
plt.xlabel("Predicted Class")
plt.ylabel("True Class")
plt.show()
