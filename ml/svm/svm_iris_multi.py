import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler

# 데이터 준비


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
    return train_test_split(
        x, y, test_size=0.2, random_state=2022)


x_train, x_test, y_train, y_test = get_iris()

# print(x_train.shape, x_test.shape)
# print(y_train.shape, y_test.shape)

# 정규화
scaler = MinMaxScaler()
# scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
y_train = y_train.values

# 학습
names = ["svc-lin", "lin-svc", "poly", "rbf", "knn"]

models = [
    SVC(kernel="linear", C=1),
    LinearSVC(C=1, max_iter=1000),
    SVC(kernel="poly", degree=3),
    SVC(kernel="rbf", C=1, gamma=0.7),
    KNeighborsClassifier(n_neighbors=5)
]

scores = []

for name, model in zip(names, models):
    model.fit(x_train, y_train)
    s = model.score(x_train, y_train)
    print(name, s)
    scores.append(s)

plt.plot(names, scores)
plt.show()

# Grid search
param_range = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
params = [
    {
        "C": param_range,
        "gamma": param_range,
        "kernel": ["rbf"]
    },
    {
        "C": param_range,
        "kernel": ["linear"]
    },
    {
        "C": param_range,
        "degree": [2, 3],
        "kernel": ["poly"]
    },
]

clf = SVC(random_state=2022)
gs = GridSearchCV(estimator=clf, scoring="accuracy",
                  cv=3, param_grid=params, n_jobs=-1, verbose=3)
gs.fit(x_train, y_train)


# 최적
print(gs.best_estimator_)
print(gs.best_score_)
print(gs.best_params_)


# 랜덤
rs = RandomizedSearchCV(estimator=clf, scoring="accuracy",
                        cv=3, param_distributions=params, n_jobs=-1, verbose=3)
rs.fit(x_train, y_train)

# 최적
print(rs.best_estimator_)
print(rs.best_score_)
print(rs.best_params_)

# 재학습
clf = SVC(C=1000, gamma=0.01, kernel="rbf")
clf.fit(x_train, y_train)

# 평가
print(clf.score(x_train, y_train))

# 최종
x_test = scaler.transform(x_test)
y_test = y_test.values

print(clf.score(x_test, y_test))

# 지표
y_pred = clf.predict(x_test)


def print_score(y_true, y_pred, average="binary"):
    acc = accuracy_score(y_true, y_pred)
    pre = precision_score(y_true, y_pred, average=average)
    rec = recall_score(y_true, y_pred, average=average)

    print("accuracy: ", acc)
    print("precision: ", pre)
    print("recall: ", rec)


print_score(y_test, y_pred, average="macro")


def plot_confusion_matrix(y_true, y_pred):
    cfm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(5, 5))
    sns.heatmap(cfm, annot=True, cbar=False, fmt="d")
    plt.xlabel("Predicted Class")
    plt.ylabel("True Class")
    plt.show()


plot_confusion_matrix(y_test, y_pred)
