import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import RobustScaler
from sklearn.svm import SVC
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier

# Data prep
df = pd.read_csv("ml/ensemble/creditcard.csv")

# x, y 분리
x = df.drop(["Time", "Class"], axis=1)
y = df["Class"]

# 학습셋 분리
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, stratify=y, random_state=2022)

# 정규화
scaler = RobustScaler()
amount = scaler.fit_transform(
    x_train["Amount"].values.reshape(-1, 1))  # 특정 column만 표준화하는 경우
x_train["Amount"] = amount.reshape(-1)

# Make similar samples with SMOTE
smote = SMOTE(k_neighbors=5)
x_train_o, y_train_o = smote.fit_resample(x_train, y_train)

# 학습
# clf = XGBClassifier(n_estimators=300, max_depth=4,
#                     learning_rate=0.1, random_state=2022)
# clf.fit(x_train_o, y_train_o)

# clf = LGBMClassifier(n_estimators=300, max_depth=4,
#                      learning_rate=0.1, random_state=2022)
# clf.fit(x_train_o, y_train_o)

clf = RandomForestClassifier(n_estimators=300, max_depth=4, random_state=2022)
clf.fit(x_train_o, y_train_o)


# 정규화
amount = scaler.transform(
    x_test["Amount"].values.reshape(-1, 1))  # 특정 column만 표준화하는 경우
x_test["Amount"] = amount.reshape(-1)

# Score
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
