import pandas as pd


# Load dataset
df = pd.read_csv("data/transactions.csv")


# Basic information
print("Dataset Shape:")
print(df.shape)


print("\nColumns:")
print(df.columns.tolist())


print("\nFirst 5 rows:")
print(df.head())


print("\nMissing Values:")
print(df.isnull().sum())


print("\nFraud Distribution:")
print(df["Fraud"].value_counts())


print("\nFraud Percentage:")
print(df["Fraud"].value_counts(normalize=True) * 100)