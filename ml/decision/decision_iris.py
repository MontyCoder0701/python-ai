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
from sklearn.tree import DecisionTreeClassifier, plot_tree


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

# 정규화 불필요

clf = DecisionTreeClassifier(random_state=2022)
clf.fit(x_train, y_train)
print(clf.score(x_train, y_train))

# Grid search
params = {
    "min_samples_leaf": range(1, 10),
    "max_depth": range(3, 10),
    "min_samples_split": range(3, 10)
}

clf = DecisionTreeClassifier(random_state=2022)
# gs = GridSearchCV(clf, params, cv=3, verbose=3)
# gs.fit(x_train, y_train)
# print(gs.best_params_)
# print(gs.best_score_)

clf = DecisionTreeClassifier(
    max_depth=3, min_samples_leaf=1, min_samples_split=3)
clf.fit(x_train, y_train)

plot_tree(clf)
plt.show()
