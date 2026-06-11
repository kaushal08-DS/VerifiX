import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Fraud Detection System",
    layout="wide"
)

st.title("💳 Fraud Detection System")

st.markdown("""
            This application uses a Random Forest model trained on credit card transactions
            to identify potentially fraudulent transactions and estimate fraud probability.
            """)

# Load model and scaler
model = joblib.load("model/fraud_model.pkl")
scaler = joblib.load("model/scaler.pkl")

uploaded_file = st.file_uploader(
    "Upload Transaction CSV",
    type=["csv"]
)

if uploaded_file:

    # Read CSV
    df = pd.read_csv(uploaded_file)

    st.write(f"Processing {len(df)} rows...")

    st.subheader("Uploaded Data")
    st.dataframe(df.head())

    # Remove target column if present
    if "Class" in df.columns:
        df = df.drop("Class", axis=1)

    # Expected columns from training dataset
    expected_columns = [
        'Time','V1','V2','V3','V4','V5','V6','V7','V8','V9',
        'V10','V11','V12','V13','V14','V15','V16','V17',
        'V18','V19','V20','V21','V22','V23','V24','V25',
        'V26','V27','V28','Amount'
    ]

    # Validate uploaded dataset
    if list(df.columns) != expected_columns:
        st.error(
            "Please upload the Credit Card Fraud dataset with columns Time, V1-V28 and Amount."
        )
        st.stop()

    with st.spinner("🔍 Analyzing transactions... Please wait."):

        # Scale features
        X = scaler.transform(df)

        # Predictions
        predictions = model.predict(X)

        # Probabilities
        probabilities = model.predict_proba(X)

    st.success("✅ Analysis Complete!")

    # Add results to dataframe
    results_df = df.copy()

    results_df["Prediction"] = predictions
    results_df["Fraud_Probability"] = probabilities[:, 1]

    # Metrics
    fraud_count = (predictions == 1).sum()

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Total Transactions",
            len(results_df)
        )

    with col2:
        st.metric(
            "Fraud Transactions Detected",
            fraud_count
        )

    st.subheader("Fraud Distribution")
    
    fig, ax = plt.subplots(figsize=(4, 3))

    results_df["Prediction"].value_counts().plot(
        kind="bar",
        ax=ax
    )

    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.pyplot(fig)

    # Results table
    st.subheader("Prediction Results")

    st.dataframe(
        results_df.head(20)
    )

    # Show only frauds
    fraud_df = results_df[
        results_df["Prediction"] == 1
    ]

    st.subheader("Detected Fraud Transactions")

    if len(fraud_df) > 0:
        st.dataframe(fraud_df)
    else:
        st.success("No fraud transactions detected.")

    csv = results_df.to_csv(index=False)

    st.download_button(
        label="📥 Download Results",
        data=csv,
        file_name="fraud_detection_results.csv",
        mime="text/csv"
    )

    results_df = results_df.sort_values(
        by="Fraud_Probability",
        ascending=False
    )

    st.dataframe(
        results_df[
            ["Prediction", "Fraud_Probability"]
        ].head(20)
    )