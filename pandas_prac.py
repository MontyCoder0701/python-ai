import pandas as pd
import numpy as np

sr = pd.Series([1, 2, 3, 4, 5], name="Apple", index=["a", "b", "c", "d", "e"])
# print(sr)
# print(sr.index)
# print(type(sr))
# print(sr[1:3])

# print(sr.values)
# print(type(sr.values))
# print(sr.reset_index())

sr = pd.Series([1, np.nan, 3, np.nan, 5])
print(sr)
print(sr[[0, 2, 4]])  # fancy indexing

idx = [False, True, True, False, False]  # boolean indexing
print(sr[sr > 3])

# Missing value
print(sr[sr.isna()])  # isnull()
print(sr.isna().sum())

x = sr.copy()
x[x.isna()] = x.mean()
print(x)

w = sr.copy()
w.fillna(w.mean())
print(w)

print(sr.dropna())
y = sr.copy()
y = y.dropna()
print(y)

z = sr.copy()
z.dropna(inplace=True)
print(z)

# Slicing
sr = pd.Series([1, 2, 3, 4, 5], index=["a", "b", "c", "d", "e"])
print(sr[1:4])
print(sr["a":"b"])
print(sr[::-1])

print(sr.sort_values(ascending=False)[:3])
print(sr.sort_index(ascending=False)[:3])

# Dataframe manipulation
doc = [['Joe', 20, 85.10, 'A', 'Swimming'],
       ['Nat', 21, 77.80, 'B', 'Reading'],
       ['Harry', 19, 91.54, 'A', 'Music'],
       ['Sam', 20, 88.78, 'A', 'Painting'],
       ['Monica', 22, 60.55, 'B', 'Dancing']]

c_name = ['Name', 'Age', 'Marks', 'Grade', 'Hobby']
idx = ['s1', 's2', 's3', 's4', 's5']

df = pd.DataFrame(doc, columns=c_name, index=idx)
print(df)
print(df.shape)
print(df.head())  # 제일 위 5개

# Dictionary
doc = {'Name': ['Joe', 'Nat', 'Harry', 'Sam', 'Monica',],
       'Age': [20, 21, 19, 20, 22],
       'Marks': [85.10, 77.80, 91.54, 88.78, 60.55],
       'Grade': ['A', 'B', 'A', 'A', 'B',],
       'Hobby': ['Swmming', 'Reading', 'Music', 'Painting', 'Dancing']}
df = pd.DataFrame(doc)
print(df)
print(df.shape)
print(df.head(3))  # 제일 위 3개
print(df.index)

x = df.dtypes
print(x)
print(x['Name'])

print(df.columns[[0, 2, 3]])
print(df.info())

# Dictionary manipulation
doc = {'Name': ['Joe', np.nan, 'Harry', np.nan, 'Monica',],
       'Age': [20, 21, 19, 20, 22],
       'Marks': [85.10, 77.80, np.nan, 88.78, np.nan],
       'Grade': ['A', 'B', 'A', 'A', 'B',],
       'Hobby': ['Swmming', 'Reading', 'Music', 'Painting', 'Dancing']}
df = pd.DataFrame(doc)
print(df.head())
print(df.info())

# Column manipulation
doc = {'Name': ['Joe', np.nan, 'Harry', np.nan, 'Monica',],
       'Age': [20, 21, 19, 20, 22],
       'Marks': [85.10, 77.80, np.nan, 88.78, np.nan],
       'Grade': ['A', 'B', 'A', 'A', 'B',],
       'Hobby': ['Swmming', 'Reading', 'Music', 'Painting', 'Dancing']}
df = pd.DataFrame(doc)
print(df.columns)
print(df[["Name", "Age"]])
print(df.rename(columns={"Marks": "Score", "Hobby": "Etc"}))

# 파일 입출력
df = pd.read_csv("doc_na.csv", index_col=0, na_values=["?", "*", "-"])
print(df)

# Titanic
titanic = pd.read_csv("titanic.csv")
print(titanic.shape)
print(titanic.head())

df = titanic.copy()
df.columns = [c.lower() for c in df.columns]
print(df)
print(df.info())
print(df.describe().T)
print(df["embarked"].unique())

# 개수 확인
print(df["embarked"].value_counts())
print(df["sex"].value_counts())
print(df["pclass"].value_counts())
print(df["survived"].value_counts())

print(df.isnull().sum(axis=0))

# Slicing, index
print(df.loc[5:10, ["pclass", "name", "survived"]])
print(df.iloc[5:10, [0, 3, 5]])

# 조건 처리
print(df["age"].min(), df["age"].max())
print(df.loc[df["age"] < 30, ['name', 'age']].count())
print(df.loc[(df["age"] < 30) & (df["sex"] == "male")].count())

# 결측치 처리 (age가 NaN인 데이터만 추출)
print(df.loc[df["age"].isnull()])

# 1. 어느 항구 제일 많은지
print(df["embarked"].value_counts())

# 2. embarked 결측치 채워넣기
df["embarked"] = df["embarked"].fillna('S')
print(df["embarked"].isnull().sum(axis=0))

# 결측치 너무 많거나 불필요한 Column 제거
df = df.drop(["cabin", "ticket", "passengerid", "name"], axis=1)
print(df.head())

df["family"] = df["sibsp"] + df["parch"]
print(df)

# 3. age 결측치 채워넣기
df["age"] = df["age"].fillna(df["age"].median())
print(df["age"])

# 그룹 함수 (map: 컬럼 단위로, apply)


def sex(x):
    if x == "male":
        return 1
    else:
        return 0


def survived(x):
    if x == "lost":
        return 0
    else:
        return 1


def pclass(x):
    if x == "1st":
        return 0
    elif x == "2nd":
        return 1
    else:
        return 2


def embarked(x):
    if x == "C":
        return 0
    elif x == "S":
        return 1
    else:
        return 2


df["sex"] = df['sex'].apply(sex)
df["survived"] = df['survived'].apply(survived)
df["pclass"] = df['pclass'].apply(pclass)
df["embarked"] = df['embarked'].apply(embarked)

print(df)

# Groupby
print(df.groupby(["sex", "pclass"])["age"].mean())

# 선실 등급별 남녀 생존자수
print(df.groupby("pclass")["sex"].value_counts())

np.savez("my_data.npz")
df.to_csv("my_data.csv")
