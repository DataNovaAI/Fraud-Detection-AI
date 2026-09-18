import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier


# Load dataset
df = pd.read_csv("data/transactions.csv")


# Features and target
X = df.drop(columns=["Transaction_ID", "Fraud"])
y = df["Fraud"]


# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# Create model
model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    class_weight="balanced"
)


# Train model
model.fit(X_train, y_train)


# Get feature importance
importance = model.feature_importances_


# Create result table
feature_importance_df = pd.DataFrame({
    "Feature": X.columns,
    "Importance": importance
})


# Sort by importance
feature_importance_df = feature_importance_df.sort_values(
    by="Importance",
    ascending=False
)


# Display results
print("Feature Importance:")
print(feature_importance_df.to_string(index=False))
import matplotlib.pyplot as plt


# Create feature importance chart
plt.figure(figsize=(10, 6))

plt.barh(
    feature_importance_df["Feature"],
    feature_importance_df["Importance"]
)

plt.xlabel("Importance")
plt.ylabel("Feature")
plt.title("Fraud Detection - Feature Importance")

plt.gca().invert_yaxis()

plt.tight_layout()

import os

output_path = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "outputs",
    "feature_importance.png"
)

plt.savefig(
    output_path,
    dpi=300
)

print(f"\nChart saved to:")
print(output_path)
