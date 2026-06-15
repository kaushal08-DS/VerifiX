import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(
    page_title="Fraud Detection System",
    layout="wide"
)

with st.sidebar:

    st.image(
        "https://cdn-icons-png.flaticon.com/512/2092/2092063.png",
        width=100
    )

    st.title("VerifiX")

    st.markdown("""
    ### Features

    ✅ Fraud Detection

    ✅ Fraud Probability

    ✅ CSV Upload

    ✅ Download Results

    ✅ Analytics Dashboard
    """)

st.markdown("""
<style>

.stApp {
    background-color: #0E1117;
    color: white;
}

h1 {
    text-align: center;
    color: #00E5FF;
}

[data-testid="stMetric"] {
    background-color: #1E222A;
    padding: 15px;
    border-radius: 10px;
    border: 1px solid #00E5FF;
}

div[data-testid="stFileUploader"] {
    background-color: #1E222A;
    padding: 15px;
    border-radius: 10px;
}

</style>
""", unsafe_allow_html=True)

st.markdown("""
<h1 style='text-align:center; color:#00E5FF;'>
💳 VerifiX - Fraud Detection System
</h1>

<h4 style='text-align:center; color:gray;'>
AI-Powered Fraud Detection using Random Forest & Machine Learning
</h4>
""", unsafe_allow_html=True)

st.markdown("""
            This application uses a Random Forest model trained on credit card transactions
            to identify potentially fraudulent transactions and estimate fraud probability.
            """)

# Load model and scaler
model = joblib.load("model/fraud_model.pkl")
scaler = joblib.load("model/scaler.pkl")

# Cache predictions
@st.cache_data
def process_data(df):
    X = scaler.transform(df)
    predictions = model.predict(X)
    probabilities = model.predict_proba(X)
    return predictions, probabilities

uploaded_file = st.file_uploader(
    "Upload Transaction CSV",
    type=["csv"]
)

if uploaded_file:

    # Read CSV
    df = pd.read_csv(uploaded_file)

    if len(df) > 50000:
        st.warning(
            "⚠️ Large dataset detected. Analysis may take 1-3 minutes depending on file size."
        )

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

    # Risk Level Function
    def risk_level(prob):
        if prob >= 0.8:
            return "🔴 High Risk"
        elif prob >= 0.5:
            return "🟠 Medium Risk"
        else:
            return "🟢 Low Risk"

    # Create Risk Level Column
    results_df["Risk_Level"] = results_df["Fraud_Probability"].apply(risk_level)

    # Sort by fraud probability
    results_df = results_df.sort_values(
        by="Fraud_Probability",
        ascending=False
    )

    # Metrics
    fraud_count = (predictions == 1).sum()
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "Transactions",
            len(results_df)
        )

    with col2:
        st.metric(
            "Frauds Detected",
            fraud_count
        )

    with col3:
        st.metric(
            "Fraud Rate",
            f"{(fraud_count/len(results_df))*100:.2f}%"
        )

    # Fraud Summary Report
    st.subheader("📋 Fraud Summary Report")

    st.info(f"""
            Total Transactions: {len(results_df)}
            
            Frauds Detected: {fraud_count}
            
            Fraud Rate: {(fraud_count/len(results_df))*100:.2f}%

    Highest Fraud Probability:
        {results_df['Fraud_Probability'].max():.2%}
    """)

    # Analytics Dashboard
    st.subheader("📊 Analytics Dashboard")

    col1, col2 = st.columns(2)

    # Chart 1
    with col1:

        st.markdown("### Fraud Distribution")

        pie_data = results_df["Prediction"].replace({
            0: "Legitimate",
            1: "Fraud"
        })

        fig, ax = plt.subplots()

        pie_data.value_counts().plot(
            kind="pie",
            autopct="%1.1f%%",
            startangle=90,
            ax=ax
        )

        ax.set_ylabel("")

        st.pyplot(fig)

    # Chart 2
    with col2:

        st.markdown("### Fraud Probability Distribution")

        fig, ax = plt.subplots()

        ax.hist(
            results_df["Fraud_Probability"],
            bins=20
        )

        st.pyplot(fig)

    # Feature Importance
    st.subheader("🔥 Feature Importance")

    importance = model.feature_importances_

    importance_df = pd.DataFrame({
        "Feature": expected_columns,
        "Importance": importance
    })

    importance_df = importance_df.sort_values(
        "Importance",
        ascending=False
    )

    st.dataframe(importance_df.head(10))

    # Fraud Heatmap
    st.subheader("🌡️ Correlation Heatmap")

    fig, ax = plt.subplots(figsize=(12, 8))

    correlation_matrix = df.corr()

    sns.heatmap(
        correlation_matrix,
        annot=False,
        cmap="coolwarm",
        linewidths=0.5,
        square=True,
        ax=ax
    )

    plt.title("Feature Correlation Matrix")

    st.pyplot(fig)

    # Top Suspicious Transactions
    st.subheader("🚨 Top 10 Suspicious Transactions")

    top_suspicious = results_df.sort_values(
        by="Fraud_Probability",
        ascending=False
    ).head(10)

    st.dataframe(
        top_suspicious[
            [
                "Fraud_Probability",
                "Risk_Level",
                "Prediction"
            ]
        ]
    )

    st.warning(
        f"Highest Fraud Probability Detected: "
        f"{top_suspicious['Fraud_Probability'].max():.2%}"
    )

    # Search by Risk Level
    st.subheader("🔍 Search Transactions")

    risk_filter = st.selectbox(
        "Select Risk Level",
        [
            "All",
            "🔴 High Risk",
            "🟠 Medium Risk",
            "🟢 Low Risk"
        ]
    )

    if risk_filter == "All":
        filtered_df = results_df
    else:
        filtered_df = results_df[
            results_df["Risk_Level"] == risk_filter
        ]

    st.dataframe(filtered_df.head(20))

    # Results table
    st.subheader("Prediction Results")

    st.dataframe(
        results_df[
            ["Prediction", "Fraud_Probability"]
        ].head(20)
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

    st.dataframe(
        results_df[
            ["Prediction", "Fraud_Probability"]
        ].head(20)
    )