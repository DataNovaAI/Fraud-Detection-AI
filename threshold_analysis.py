import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score


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


# Fraud probabilities
fraud_probability = model.predict_proba(X_test)[:, 1]


# Test different thresholds
thresholds = [0.50, 0.40, 0.30, 0.20]


for threshold in thresholds:

    y_pred = (
        fraud_probability >= threshold
    ).astype(int)

    precision = precision_score(
        y_test,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        y_pred,
        zero_division=0
    )

    print(f"\nThreshold: {threshold}")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1-score: {f1:.2f}")