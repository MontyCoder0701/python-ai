import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.metrics import confusion_matrix
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import VotingClassifier

iris = datasets.load_iris()
# print(iris.target_names)  # iris["target_names"]
# print(iris.feature_names)
# print(iris.data)

# Df로 변환
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df["target"] = iris.target

# 결측치, 중복치 확인
# print(df.isna().sum())
# print(df.duplicated().sum())

# x,y 분리
x = df.drop(['target'], axis=1)
y = df['target']

# 학습셋 분리
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, stratify=y, random_state=2022)

# 정규화 (Tree 기반인 경우 불필요)
scaler = StandardScaler()
# scaler.fit(x_train)
# x_train = scaler.transform(x_train)
x_train = scaler.fit_transform(x_train)

# Voting
clf1 = KNeighborsClassifier()
clf2 = SVC()
clf3 = DecisionTreeClassifier()

clf = VotingClassifier(
    estimators=[("knn", clf1), ("svc", clf2), ("tree", clf3)],
    voting="hard",
    weights=[1, 1, 1]
)

clf.fit(x_train, y_train)

# 정확도 판정
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
