import pandas as pd
import matplotlib
import os
import matplotlib.pyplot as plt
import seaborn as sns
print("Understanding Data")

file_name='sales_data.csv'
if not os.path.exists(file_name):
    print(f"Error: File'{file_name}' not found")
    exit()

df=pd.read_csv(file_name)
print("Successfully Loaded")
print(f"shape of the Dataset: {df.shape}")

print(df.head())
print(df.info())
print(df.describe())

print("Handling missing values:")

print(df.isnull().sum())

#With using median
median_age=df["Age"].median()
df["Age"]=df["Age"].fillna(median_age)
print(median_age)

median_spending=df["Spending"].median()
df["Spending"]=df["Spending"].fillna(median_spending)
print(median_spending)

# Distribution Analysis

plt.figure(figsize=(7,4))
df["Spending"].hist(bins=10,color="skyblue",edgecolor="black")
plt.title("Distribution of Spending")
plt.xlabel("Spending Amount")
plt.ylabel("Number of Customers")


# Correlational Matrix

correlation = df.select_dtypes(include='number').corr()
print(correlation)


print("Ploating Correlation Heatmap")
plt.figure(figsize=(7,4))
sns.heatmap(correlation, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap")
plt.show()

# Outliers Detection

print("Find the Outliers in age")
outliers=df[df['Age']>100]
print("Found Outliers (s):")
print(outliers)