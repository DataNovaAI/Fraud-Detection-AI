import pandas as pd
import joblib


# Load trained model
model = joblib.load(
    "models/fraud_detection_model.pkl"
)


# Create a sample transaction
transaction = pd.DataFrame([{
    "Amount": 4500,
    "Hour": 3,
    "Customer_Age": 35,
    "Previous_Transactions": 10,
    "Distance_From_Last_Transaction": 850,
    "Device_Changed": 1,
    "International": 1
}])


# Predict fraud probability
fraud_probability = model.predict_proba(
    transaction
)[0][1]


print(
    f"Fraud Probability: "
    f"{fraud_probability * 100:.2f}%"
)


# Prediction using default threshold
prediction = (
    fraud_probability >= 0.5
)


if prediction:
    print("Prediction: FRAUD")
else:
    print("Prediction: NORMAL")