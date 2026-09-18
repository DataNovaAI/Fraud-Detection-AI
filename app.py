import streamlit as st
import pandas as pd
import joblib

# Title
st.set_page_config(
    page_title="Fraud Detection AI",
    page_icon="🛡️",
    layout="wide"
)

st.title("🛡️ Fraud Detection AI")
st.caption(
    "AI-powered transaction risk analysis and fraud detection"
)
with st.sidebar:

    st.header("🛡️ About the Model")

    st.write(
        "This fraud detection system uses "
        "Machine Learning to identify "
        "potentially fraudulent transactions."
    )

    st.divider()

    st.subheader("Model Information")

    st.write("Model: Random Forest")
    st.write("Training Transactions: 10,000")
    st.write("Fraud Rate: 2.00%")

    st.divider()

    st.subheader("Key Features")

    st.write("• Transaction Amount")
    st.write("• Transaction Hour")
    st.write("• Device Change")
    st.write("• International Transaction")
    st.write("• Distance From Last Transaction")
# Load trained model
model = joblib.load(
    "models/fraud_detection_model.pkl"
)


# Page configuration
st.set_page_config(
    page_title="Fraud Detection AI",
    page_icon="🛡️",
    layout="centered"
)

# Load transaction data
df = pd.read_csv(
    "data/transactions.csv"
)

total_transactions = len(df)

fraud_transactions = df["Fraud"].sum()

fraud_rate = (
    fraud_transactions / total_transactions
)

average_amount = df["Amount"].mean()


# Dashboard KPIs
st.header("📊 Transaction Overview")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Total Transactions",
        f"{total_transactions:,}"
    )

with col2:
    st.metric(
        "Fraud Transactions",
        f"{fraud_transactions:,}"
    )

with col3:
    st.metric(
        "Fraud Rate",
        f"{fraud_rate * 100:.2f}%"
    )

with col4:
    st.metric(
        "Average Amount",
        f"${average_amount:,.2f}"
    )
# Fraud distribution chart
st.subheader("📈 Fraud Distribution")

fraud_counts = df["Fraud"].value_counts()

chart_data = pd.DataFrame({
    "Transaction Type": ["Normal", "Fraud"],
    "Count": [
        fraud_counts.get(0, 0),
        fraud_counts.get(1, 0)
    ]
})

st.bar_chart(
    chart_data.set_index("Transaction Type")
)
# Feature importance
st.subheader("🎯 Fraud Detection Factors")

feature_importance = pd.DataFrame({
    "Feature": model.feature_names_in_,
    "Importance": model.feature_importances_
})

feature_importance = feature_importance.sort_values(
    "Importance",
    ascending=False
)

st.bar_chart(
    feature_importance.set_index("Feature")
)
# Upload transaction file
st.divider()

st.header("📂 Analyze Transaction File")

uploaded_file = st.file_uploader(
    "Upload a CSV file",
    type=["csv"]
)

if uploaded_file is not None:

    uploaded_df = pd.read_csv(
        uploaded_file
    )

    st.success(
        "File uploaded successfully!"
    )

    st.write(
        f"Number of transactions: "
        f"{len(uploaded_df):,}"
    )

    required_columns = [
        "Amount",
        "Hour",
        "Customer_Age",
        "Previous_Transactions",
        "Distance_From_Last_Transaction",
        "Device_Changed",
        "International"
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in uploaded_df.columns
    ]

    if missing_columns:

        st.error(
            "Missing required columns: "
            + ", ".join(missing_columns)
        )

    else:

        X_uploaded = uploaded_df[
            required_columns
        ]

        fraud_probabilities = (
            model.predict_proba(X_uploaded)[:, 1]
        )

        uploaded_df["Fraud_Probability"] = (
            fraud_probabilities
        )

        threshold = st.slider(
            "Fraud Detection Threshold",
            min_value=0.10,
            max_value=0.90,
            value=0.50,
            step=0.05
        )

        uploaded_df["Prediction"] = (
                fraud_probabilities >= threshold
        ).astype(int)
        uploaded_df["Prediction_Label"] = (
            uploaded_df["Prediction"]
            .map({
                0: "Normal",
                1: "Fraud"
            })
        )


        def risk_level(probability):

            if probability >= 0.70:
                return "High Risk"

            elif probability >= 0.30:
                return "Medium Risk"

            else:
                return "Low Risk"


        uploaded_df["Risk_Level"] = (
            uploaded_df["Fraud_Probability"]
            .apply(risk_level)
        )

        fraud_count = (
            uploaded_df["Prediction"].sum()
        )

        normal_count = (
            len(uploaded_df) - fraud_count
        )

        fraud_rate_uploaded = (
            fraud_count / len(uploaded_df)
        )
        risk_counts = uploaded_df["Risk_Level"].value_counts()

        high_risk = risk_counts.get("High Risk", 0)
        medium_risk = risk_counts.get("Medium Risk", 0)
        low_risk = risk_counts.get("Low Risk", 0)

        st.subheader("⚠️ Risk Summary")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "🔴 High Risk",
                f"{high_risk:,}"
            )

        with col2:
            st.metric(
                "🟠 Medium Risk",
                f"{medium_risk:,}"
            )

        with col3:
            st.metric(
                "🟢 Low Risk",
                f"{low_risk:,}"
            )
        st.subheader(
            "📊 File Analysis Results"
        )

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "Total Transactions",
                f"{len(uploaded_df):,}"
            )

        with col2:
            st.metric(
                "Fraud Detected",
                f"{fraud_count:,}"
            )

        with col3:
            st.metric(
                "Fraud Rate",
                f"{fraud_rate_uploaded * 100:.2f}%"
            )

        st.subheader(
            "🔍 Transaction Predictions"
        )

        sorted_df = uploaded_df.sort_values(
            "Fraud_Probability",
            ascending=False
        )

        st.dataframe(
            sorted_df
        )
        st.subheader("🚨 High Risk Transactions")

        high_risk_df = uploaded_df[
            uploaded_df["Risk_Level"] == "High Risk"
            ]

        st.write(
            f"High-risk transactions: "
            f"{len(high_risk_df):,}"
        )

        st.dataframe(
            high_risk_df
        )
        st.download_button(
            label="📥 Download Analysis Results",
            data=sorted_df.to_csv(index=False),
            file_name="fraud_analysis_results.csv",
            mime="text/csv"
        )



# Input section
st.header("Transaction Information")

amount = st.number_input(
    "Transaction Amount",
    min_value=0.0,
    value=1000.0
)

hour = st.number_input(
    "Transaction Hour",
    min_value=0,
    max_value=23,
    value=12
)

customer_age = st.number_input(
    "Customer Age",
    min_value=18,
    max_value=100,
    value=35
)

previous_transactions = st.number_input(
    "Previous Transactions",
    min_value=1,
    value=10
)

distance = st.number_input(
    "Distance From Last Transaction",
    min_value=0.0,
    value=100.0
)

device_changed = st.selectbox(
    "Device Changed?",
    ["No", "Yes"]
)

international = st.selectbox(
    "International Transaction?",
    ["No", "Yes"]
)


# Convert Yes/No to 0/1
device_changed_value = (
    1 if device_changed == "Yes" else 0
)

international_value = (
    1 if international == "Yes" else 0
)


# Create transaction dataframe
transaction = pd.DataFrame([{
    "Amount": amount,
    "Hour": hour,
    "Customer_Age": customer_age,
    "Previous_Transactions": previous_transactions,
    "Distance_From_Last_Transaction": distance,
    "Device_Changed": device_changed_value,
    "International": international_value
}])


# Prediction
if st.button("🔍 Analyze Transaction"):

    fraud_probability = model.predict_proba(
        transaction
    )[0][1]

    probability_percent = (
        fraud_probability * 100
    )

    st.divider()

    st.header("Fraud Analysis")

    # Show probability
    st.metric(
        "Fraud Probability",
        f"{probability_percent:.2f}%"
    )

    # Probability progress bar
    st.progress(
        float(fraud_probability)
    )

    st.write("Risk Level")

    if fraud_probability >= 0.5:

        st.error(
            "🚨 HIGH RISK — FRAUD DETECTED"
        )

        st.write(
            "This transaction shows a high "
            "probability of fraudulent activity."
        )

    else:

        st.success(
            "✅ LOW RISK — TRANSACTION APPEARS NORMAL"
        )
    if fraud_probability >= 0.5:

            st.warning(
                "⚠️ Recommended Action: "
                "Review this transaction manually "
                "before approving it."
            )

    else:
        st.info(
                "ℹ️ Recommended Action: "
                "Transaction can proceed with normal monitoring."
        )

        st.write(
            "This transaction does not show "
            "a high probability of fraud."
        )

    st.divider()

    # Transaction summary
    st.subheader("🎯 Potential Risk Factors")

    risk_factors = []

    if transaction["Amount"].iloc[0] >= 3000:
        risk_factors.append(
            "💰 High transaction amount"
        )

    if transaction["Hour"].iloc[0] < 5:
        risk_factors.append(
            "🌙 Unusual transaction hour"
        )

    if transaction["Device_Changed"].iloc[0] == 1:
        risk_factors.append(
            "📱 Device changed"
        )

    if transaction["International"].iloc[0] == 1:
        risk_factors.append(
            "🌍 International transaction"
        )

    if transaction[
        "Distance_From_Last_Transaction"
    ].iloc[0] >= 500:
        risk_factors.append(
            "📍 Large distance from previous transaction"
        )

    if risk_factors:

        for factor in risk_factors:
            st.write(factor)

    else:

        st.success(
            "No notable risk factors detected."
        )

    col1, col2 = st.columns(2)

    with col1:
        st.write(f"💰 Amount: ${amount:,.2f}")
        st.write(f"🕐 Hour: {hour}")
        st.write(f"👤 Customer Age: {customer_age}")
        st.write(
            f"🔄 Previous Transactions: "
            f"{previous_transactions}"
        )

    with col2:
        st.write(
            f"📍 Distance: {distance:,.2f}"
        )
        st.write(
            f"📱 Device Changed: "
            f"{device_changed}"
        )
        st.write(
            f"🌍 International: "
            f"{international}"
        )