import seaborn as sns
import matplotlib.pyplot as plt

df = sns.load_dataset("penguins")
print(df["species"].unique())
print(df["island"].unique())

print(df.groupby("species")["island"].value_counts())

# sns.scatterplot(data=df, x="flipper_length_mm",
#                 y="bill_length_mm", hue="species")

# sns.histplot(data=df, x="species", hue="species")

fig, axes = plt.subplots(2, 2)
sns.scatterplot(data=df, x="flipper_length_mm",
                y="bill_length_mm", hue="species", ax=axes[0, 0])

sns.scatterplot(data=df, x="flipper_length_mm",
                y="bill_length_mm", hue="species", ax=axes[0, 1])

sns.histplot(data=df, x="species", hue="species", ax=axes[1, 0])
sns.histplot(data=df, x="species", hue="species", ax=axes[1, 1])

sns.displot(data=df, x="flipper_length_mm",
            kind="hist", hue="species", col="species", height=3, aspect=1)

sns.displot(data=df, x="flipper_length_mm",
            kind="kde", hue="species", col="species", height=3, aspect=1)

sns.relplot(data=df, x="flipper_length_mm", y="bill_length_mm",
            kind="scatter", hue="species", col="species", height=3, aspect=1)

plt.show()
