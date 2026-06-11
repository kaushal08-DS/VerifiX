import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load Dataset
df = pd.read_csv("data/creditcard.csv")

# -----------------------------
# EDA
# -----------------------------

print(df["Class"].value_counts())

plt.figure(figsize=(6, 4))
sns.countplot(x="Class", data=df)
plt.title("Fraud vs Non-Fraud Transactions")
plt.show()

plt.figure(figsize=(8, 4))
sns.histplot(df["Amount"], bins=50)
plt.title("Transaction Amount Distribution")
plt.show()

# -----------------------------
# Features and Target
# -----------------------------

X = df.drop("Class", axis=1)
y = df["Class"]

print("Features Shape:", X.shape)
print("Target Shape:", y.shape)

# -----------------------------
# Train-Test Split
# -----------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# -----------------------------
# Feature Scaling
# -----------------------------

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# -----------------------------
# SMOTE
# -----------------------------

smote = SMOTE(random_state=42)

X_train_smote, y_train_smote = smote.fit_resample(
    X_train,
    y_train
)

print("\nBefore SMOTE:")
print(y_train.value_counts())

print("\nAfter SMOTE:")
print(y_train_smote.value_counts())

# -----------------------------
# Train Model
# -----------------------------

rf_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

rf_model.fit(
    X_train_smote,
    y_train_smote
)

predictions = rf_model.predict(X_test)

joblib.dump(rf_model, "model/fraud_model.pkl")
joblib.dump(scaler, "model/scaler.pkl")

print("Model Saved Successfully!")

# -----------------------------
# Predictions
# -----------------------------

predictions = rf_model.predict(X_test)

# -----------------------------
# Evaluation
# -----------------------------

accuracy = accuracy_score(
    y_test,
    predictions
)

print(f"\nAccuracy: {accuracy:.4f}")

print("\nClassification Report:")
print(classification_report(
    y_test,
    predictions
))

print("\nConfusion Matrix:")
print(confusion_matrix(
    y_test,
    predictions
))