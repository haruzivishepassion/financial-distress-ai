# ===============================
# FINANCIAL DISTRESS AI SYSTEM (PRO VERSION)
# ===============================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import os   # ✅ ADD THIS

from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet

# OPTIONAL AI
try:
    from openai import OpenAI
    client = OpenAI()
    AI_AVAILABLE = True
except:
    AI_AVAILABLE = False

# ===============================
# 🔍 API KEY TEST (TEMPORARY ONLY)
# ===============================
# st.write("DEBUG API KEY:", os.getenv("OPENAI_API_KEY"))

# LOAD MODEL
model = joblib.load("model/distress_model.pkl")
scaler = joblib.load("model/scaler.pkl")

st.set_page_config(layout="wide")
st.title("💰 Financial Distress AI System (Pro)")

file = st.file_uploader("Upload CSV or Excel", type=["csv", "xlsx"])

if file:

    # ===============================
    # LOAD DATA
    # ===============================
    if file.name.endswith(".csv"):
        df = pd.read_csv(file)
    else:
        df = pd.read_excel(file)

    st.write("Data Preview", df.head())

    if "company_name" in df.columns:
        df = df.drop(columns=["company_name"])

    df = df.fillna(df.median(numeric_only=True))

    X_scaled = scaler.transform(df)

    probs = model.predict_proba(X_scaled)[:, 1]

    # ===============================
    # CREDIT SCORE
    # ===============================
    credit_score = (1 - probs) * 100
    credit_score = np.clip(credit_score, 0, 100).round(2)

    def risk_band(score):
        if score >= 80:
            return "LOW RISK"
        elif score >= 60:
            return "MEDIUM RISK"
        elif score >= 40:
            return "HIGH RISK"
        else:
            return "VERY HIGH RISK"

    df["Credit Score"] = credit_score
    df["Risk"] = [risk_band(x) for x in credit_score]

    st.subheader("📊 Results")
    st.dataframe(df)

    # ===============================
    # 🔍 SHAP EXPLAINABILITY
    # ===============================
    st.subheader("🧠 Explainability (SHAP)")

    explainer = shap.Explainer(model)
    shap_values = explainer(X_scaled[:100])

    st.write("Feature Impact Summary")
    shap.summary_plot(shap_values, X_scaled[:100], show=False)
    st.pyplot()

    # ===============================
    # 📄 PROFESSIONAL PDF REPORT
    # ===============================
    if st.button("Generate Professional PDF"):

        doc = SimpleDocTemplate("report.pdf")
        styles = getSampleStyleSheet()

        content = []

        content.append(Paragraph("Financial Distress Report", styles["Title"]))
        content.append(Spacer(1, 12))

        for i, row in df.head(5).iterrows():
            text = f"""
            Company Index: {i}<br/>
            Credit Score: {row['Credit Score']}<br/>
            Risk Level: {row['Risk']}<br/>
            """
            content.append(Paragraph(text, styles["Normal"]))
            content.append(Spacer(1, 12))

        doc.build(content)

        with open("report.pdf", "rb") as f:
            st.download_button("Download PDF", f, file_name="report.pdf")

# ===============================
# 🤖 REAL AI CHATBOT
# ===============================
st.sidebar.title("🤖 AI Assistant")

q = st.sidebar.text_input("Ask anything about results")

if q:

    if AI_AVAILABLE:
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a financial risk analyst."},
                    {"role": "user", "content": q}
                ]
            )
            st.sidebar.write(response.choices[0].message.content)
        except:
            st.sidebar.write("AI error — fallback active.")
    else:
        # fallback
        st.sidebar.write("AI not configured. Ask about score, risk, or model.")
