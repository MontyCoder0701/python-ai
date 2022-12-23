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

df = pd.read_csv("ml/svm/train.csv")
print(df)

# 결측치
# 중복치
# print(df.isna().sum())
# print(df.duplicated().sum())

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

# clf = SVC(random_state=2022)
# gs = GridSearchCV(estimator=clf, scoring="accuracy",
#                   cv=3, param_grid=params, n_jobs=-1, verbose=3)
# gs.fit(x_train, y_train)


# # 최적
# print(gs.best_estimator_)
# print(gs.best_score_)
# print(gs.best_params_)

# Best model 찾기

clf = SVC(C=10, gamma=0.01, kernel="rbf")
clf.fit(x_train, y_train)
print(clf.score(x_train, y_train))


# 테스트파일 읽어서 score 계산하기
test = pd.read_csv("ml/svm/test.csv")

test['Activity'] = test['Activity'].map({
    'STANDING': 0,
    'SITTING': 1,
    'LAYING': 2,
    'WALKING': 3,
    'WALKING_DOWNSTAIRS': 4,
    'WALKING_UPSTAIRS': 5,
})

x_test = test.drop(['Activity'], axis=1)
y_test = test['Activity']

# 최종
x_test = scaler.transform(x_test)
y_test = y_test.values

print(x_train.shape, x_test.shape)
print(y_train.shape, y_test.shape)

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
