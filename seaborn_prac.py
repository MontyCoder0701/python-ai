import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Penguin
# df = sns.load_dataset("penguins")
# print(df["species"].unique())
# print(df["island"].unique())

# print(df.groupby("species")["island"].value_counts())

# # sns.scatterplot(data=df, x="flipper_length_mm",
# #                 y="bill_length_mm", hue="species")

# # sns.histplot(data=df, x="species", hue="species")

# fig, axes = plt.subplots(2, 2)
# sns.scatterplot(data=df, x="flipper_length_mm",
#                 y="bill_length_mm", hue="species", ax=axes[0, 0])

# sns.scatterplot(data=df, x="flipper_length_mm",
#                 y="bill_length_mm", hue="species", ax=axes[0, 1])

# sns.histplot(data=df, x="species", hue="species", ax=axes[1, 0])
# sns.histplot(data=df, x="species", hue="species", ax=axes[1, 1])

# sns.displot(data=df, x="flipper_length_mm",
#             kind="hist", hue="species", col="species", height=3, aspect=1)

# sns.displot(data=df, x="flipper_length_mm",
#             kind="kde", hue="species", col="species", height=3, aspect=1)

# sns.relplot(data=df, x="flipper_length_mm", y="bill_length_mm",
#             kind="scatter", hue="species", col="species", height=3, aspect=1)

# Titanic
# df = pd.read_csv("titanic.csv")
# print(df)

# # sns.barplot(data=df, x="Pclass", y="Fare")
# # sns.countplot(data=df, x="Pclass")

# print(df["Pclass"].value_counts())

# # sns.barplot(data=df, x="Pclass", y="Fare", hue="Sex", estimator=np.mean)
# # sns.boxplot(data=df, x="Pclass", y="Age")
# # sns.violinplot(data=df, x="Pclass", y="Age")

# data = np.random.rand(3, 3)
# print(data)
# sns.heatmap(data=data, annot=True, cmap="coolwarm")

# Iris
df = pd.read_csv("iris.csv")
print(df)

sns.pairplot(df, hue="Species")

plt.show()
