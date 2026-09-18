import pandas as pd
import numpy as np
import random


# Reproducibility
random.seed(42)
np.random.seed(42)


# Number of transactions
num_transactions = 10000


# Transaction IDs
transaction_ids = [
    f"TX_{i:06d}"
    for i in range(1, num_transactions + 1)
]


# Transaction amount
amount = np.round(
    np.random.uniform(5, 5000, num_transactions),
    2
)


# Transaction time
hour = np.random.randint(
    0,
    24,
    num_transactions
)


# Customer age
customer_age = np.random.randint(
    18,
    80,
    num_transactions
)


# Number of previous transactions
previous_transactions = np.random.randint(
    1,
    100,
    num_transactions
)


# Distance from previous transaction
distance_from_last_transaction = np.round(
    np.random.uniform(0, 1000, num_transactions),
    2
)


# Device change indicator
device_changed = np.random.choice(
    [0, 1],
    size=num_transactions,
    p=[0.85, 0.15]
)


# International transaction
international = np.random.choice(
    [0, 1],
    size=num_transactions,
    p=[0.90, 0.10]
)


# Calculate fraud score
fraud_score = (
    0.0004 * amount
    + 0.8 * device_changed
    + 0.7 * international
    + 0.001 * distance_from_last_transaction
    + 0.5 * ((hour < 5).astype(int))
)


# Select approximately 2% transactions as fraud
fraud_threshold = np.percentile(
    fraud_score,
    98
)


fraud = (
    fraud_score >= fraud_threshold
).astype(int)


# Create DataFrame
df = pd.DataFrame({
    "Transaction_ID": transaction_ids,
    "Amount": amount,
    "Hour": hour,
    "Customer_Age": customer_age,
    "Previous_Transactions": previous_transactions,
    "Distance_From_Last_Transaction": distance_from_last_transaction,
    "Device_Changed": device_changed,
    "International": international,
    "Fraud": fraud
})


# Save dataset
df.to_csv(
    "data/transactions.csv",
    index=False
)


# Display results
print("Fraud detection dataset created successfully!")

print(f"Number of transactions: {len(df)}")

print(f"Fraudulent transactions: {df['Fraud'].sum()}")

print(
    f"Fraud rate: "
    f"{df['Fraud'].mean() * 100:.2f}%"
)