import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix


# Load dataset
df = pd.read_csv("data/transactions.csv")


# Features
X = df.drop(columns=["Transaction_ID", "Fraud"])


# Target
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


# Predictions
y_pred = model.predict(X_test)


# Confusion Matrix
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))


# Classification Report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
import joblib


# Save trained model
joblib.dump(
    model,
    "models/fraud_detection_model.pkl"
)

print("\nModel saved successfully!")