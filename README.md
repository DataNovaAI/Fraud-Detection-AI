# 🛡️ Fraud Detection AI

## AI-Powered Transaction Risk Analysis

Fraud Detection AI is a Machine Learning application designed to identify potentially fraudulent financial transactions and analyze transaction risk.

The project uses a Random Forest classification model and provides an interactive Streamlit dashboard for both individual transaction analysis and batch CSV analysis.

---

## 🚀 Features

### 📊 Transaction Overview

The dashboard provides key transaction metrics:

- Total Transactions
- Fraud Transactions
- Fraud Rate
- Average Transaction Amount
- Fraud Distribution

### 🤖 Machine Learning Fraud Detection

The system uses a Random Forest Classifier to predict the probability that a transaction is fraudulent.

The model considers features such as:

- Transaction Amount
- Transaction Hour
- Customer Age
- Previous Transactions
- Distance From Last Transaction
- Device Change
- International Transaction

### 🔍 Individual Transaction Analysis

Users can enter transaction information manually and receive:

- Fraud Probability
- Risk Level
- Recommended Action
- Potential Risk Factors
- Transaction Summary

### 📂 Batch Transaction Analysis

Users can upload a CSV file and analyze multiple transactions at once.

The system provides:

- Fraud Probability for each transaction
- Fraud / Normal prediction
- High / Medium / Low risk classification
- High-risk transaction list
- Downloadable analysis results

### 🎯 Adjustable Fraud Threshold

Users can adjust the fraud detection threshold to control how sensitive the prediction system is.

This demonstrates the trade-off between:

- Precision
- Recall
- False Positives
- False Negatives

---

## 📈 Model Performance

The model was evaluated using a test set containing 2,000 transactions.

Example evaluation results at the default classification threshold:

- Accuracy: ~99%
- Fraud Precision: ~90%
- Fraud Recall: ~47%
- Fraud F1-score: ~62%

Because fraud detection is an imbalanced classification problem, accuracy alone should not be used to evaluate the model.

The project also includes threshold analysis to demonstrate how changing the classification threshold affects precision and recall.

---

## 🧠 Feature Importance

The Random Forest model provides feature importance information to show which input variables contributed most to the model's decisions.

The current synthetic dataset shows the following relative importance:

1. Amount
2. Device Changed
3. Distance From Last Transaction
4. International Transaction
5. Hour
6. Previous Transactions
7. Customer Age

These values are specific to the synthetic dataset and should not be interpreted as universal indicators of financial fraud.

---

## 🖥️ Dashboard

The application is built with Streamlit and provides an interactive interface for transaction risk analysis.

Main dashboard sections include:

- Transaction Overview
- Fraud Distribution
- Fraud Detection Factors
- Individual Transaction Analysis
- CSV File Analysis
- Risk Summary
- High-Risk Transactions
- Downloadable Analysis Results

---

## 📁 Project Structure

```text
Fraud Detection AI/
│
├── app.py
├── create_data.py
├── analyze_data.py
├── train_model.py
├── feature_importance.py
├── threshold_analysis.py
├── predict_transaction.py
│
├── data/
│   └── transactions.csv
│
├── models/
│   └── fraud_detection_model.pkl
│
├── outputs/
│   └── feature_importance.png
│
├── requirements.txt
└── .gitignore