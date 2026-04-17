# ===============================
# FINANCIAL DISTRESS AI MODEL (FINAL TUNED XGBOOST)
# ===============================

import os
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler

from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
import joblib

print("Loading dataset...")
df = pd.read_csv("data/american_bankruptcy.csv")

df = df.drop_duplicates()

if "company_name" in df.columns:
    df = df.drop(columns=["company_name"])

df["status_label"] = df["status_label"].map({
    "alive": 0,
    "failed": 1
})

df = df.fillna(df.median(numeric_only=True))

print("\nHandling outliers...")
lower = df.quantile(0.01)
upper = df.quantile(0.99)
df = df.clip(lower=lower, upper=upper, axis=1)

print("\nSplitting data...")
X = df.drop("status_label", axis=1)
y = df["status_label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("\nApplying SMOTE...")
smote = SMOTE(random_state=42)
X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)

print("\nScaling data...")
scaler = StandardScaler()
X_train_sm = scaler.fit_transform(X_train_sm)
X_test_scaled = scaler.transform(X_test)

print("\nTraining XGBoost model...")
model = XGBClassifier(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=5,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    eval_metric="logloss"
)

model.fit(X_train_sm, y_train_sm)

print("\nEvaluating model...")
y_prob = model.predict_proba(X_test_scaled)[:, 1]

# 🔥 IMPROVED THRESHOLD
threshold = 0.60
y_pred = (y_prob >= threshold).astype(int)

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# ===============================
# CREDIT SCORING SYSTEM
# ===============================
credit_score = (1 - y_prob) * 100
credit_score = np.clip(credit_score, 0, 100).round(2)

def risk_band(score):
    if score >= 80:
        return "LOW RISK 🟢"
    elif score >= 60:
        return "MEDIUM RISK 🟠"
    elif score >= 40:
        return "HIGH RISK 🔴"
    else:
        return "VERY HIGH RISK ⛔"

risk_category = pd.Series(credit_score).apply(risk_band)

results = pd.DataFrame({
    "Actual": y_test.values,
    "Default_Probability": y_prob,
    "Credit_Score": credit_score,
    "Risk_Category": risk_category
})

print("\nSample Results:")
print(results.head())

print("\nSaving model...")
os.makedirs("model", exist_ok=True)
joblib.dump(model, "model/distress_model.pkl")
joblib.dump(scaler, "model/scaler.pkl")

print("\nDONE ✅ Model ready for app")
