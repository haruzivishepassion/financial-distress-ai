# ===============================
# FINTECH AI CREDIT RISK SYSTEM (PRODUCTION VERSION)
# ===============================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import sqlite3
import matplotlib.pyplot as plt
import io
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet

# ===============================
# OPENAI (CHATBOT)
# ===============================
try:
    from openai import OpenAI
    client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
    AI_AVAILABLE = True
except:
    AI_AVAILABLE = False

# ===============================
# LOAD MODEL
# ===============================
model = joblib.load("model/distress_model.pkl")
scaler = joblib.load("model/scaler.pkl")

st.set_page_config(layout="wide")
st.title("💰 FINTECH CREDIT RISK PLATFORM (PRO)")

# ===============================
# DATABASE SETUP (SQLite)
# ===============================
conn = sqlite3.connect("credit_risk.db", check_same_thread=False)
c = conn.cursor()
c.execute("""
CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    credit_score REAL,
    risk TEXT
)
""")
conn.commit()

def save_to_db(score, risk):
    c.execute("INSERT INTO predictions (credit_score, risk) VALUES (?,?)", (score, risk))
    conn.commit()

# ===============================
# FILE UPLOAD
# ===============================
file = st.file_uploader("Upload Financial Data (CSV/Excel)", type=["csv", "xlsx"])

if file:

    # LOAD DATA
    if file.name.endswith(".csv"):
        df = pd.read_csv(file)
    else:
        df = pd.read_excel(file)

    st.subheader("📂 Raw Data")
    st.dataframe(df.head())

    # CLEAN
    if "company_name" in df.columns:
        df = df.drop(columns=["company_name"])

    df = df.fillna(df.median(numeric_only=True))

    # SCALE
    X_scaled = scaler.transform(df)

    # PREDICT
    probs = model.predict_proba(X_scaled)[:, 1]

    credit_score = (1 - probs) * 100
    credit_score = np.clip(credit_score, 0, 100).round(2)

    def risk_band(x):
        if x >= 80:
            return "LOW RISK 🟢"
        elif x >= 60:
            return "MEDIUM RISK 🟡"
        elif x >= 40:
            return "HIGH RISK 🟠"
        return "VERY HIGH RISK 🔴"

    df["Credit Score"] = credit_score
    df["Risk"] = [risk_band(x) for x in credit_score]

    # SAVE TO DB
    for s, r in zip(credit_score, df["Risk"]):
        save_to_db(s, r)

    # ===============================
    # FILTER DASHBOARD
    # ===============================
    st.subheader("📊 Filter Dashboard")

    risk_filter = st.multiselect(
        "Filter Risk Level",
        df["Risk"].unique(),
        default=df["Risk"].unique()
    )

    filtered_df = df[df["Risk"].isin(risk_filter)]
    st.dataframe(filtered_df)

    # ===============================
    # VISUALS
    # ===============================
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📊 Risk Distribution")
        fig, ax = plt.subplots()
        filtered_df["Risk"].value_counts().plot(kind="bar", ax=ax)
        st.pyplot(fig)

    with col2:
        st.subheader("📈 Credit Score Distribution")
        fig2, ax2 = plt.subplots()
        ax2.hist(filtered_df["Credit Score"], bins=20)
        st.pyplot(fig2)

    # ===============================
    # FEATURE IMPORTANCE
    # ===============================
    st.subheader("🧠 Key Drivers of Financial Distress")

    importance = model.feature_importances_
    features = df.drop(columns=["Credit Score", "Risk"], errors="ignore").columns[:len(importance)]

    fig3, ax3 = plt.subplots()
    ax3.barh(features, importance)
    st.pyplot(fig3)

    # ===============================
    # INDIVIDUAL COMPANY VIEW
    # ===============================
    st.subheader("🏢 Individual Company Analysis")

    idx = st.number_input("Select Company Index", 0, len(df)-1, 0)

    st.write(df.iloc[idx])

    st.metric("Credit Score", df.iloc[idx]["Credit Score"])
    st.write("Risk Level:", df.iloc[idx]["Risk"])

    # ===============================
    # SHAP EXPLANATION (SINGLE COMPANY)
    # ===============================
    st.subheader("🧠 Explainability (Company Level)")

    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer(X_scaled)

        st.text("Feature impact for selected company:")
        shap.force_plot(
            explainer.expected_value,
            shap_values[idx].values,
            X_scaled[idx],
            matplotlib=True,
            show=False
        )
        st.pyplot()

    except:
        st.warning("SHAP visualization not supported in this environment.")

    # ===============================
    # PDF REPORT
    # ===============================
    if st.button("Download Full Report PDF"):

        buffer = "report.pdf"
        doc = SimpleDocTemplate(buffer)
        styles = getSampleStyleSheet()

        content = []
        content.append(Paragraph("FINTECH CREDIT RISK REPORT", styles["Title"]))
        content.append(Spacer(1, 12))

        for i in range(min(10, len(df))):
            text = f"""
            Company {i}<br/>
            Credit Score: {df.iloc[i]['Credit Score']}<br/>
            Risk: {df.iloc[i]['Risk']}<br/>
            """
            content.append(Paragraph(text, styles["Normal"]))
            content.append(Spacer(1, 10))

        doc.build(content)

        with open("report.pdf", "rb") as f:
            st.download_button("⬇ Download Report", f, file_name="credit_report.pdf")

# ===============================
# 🤖 AI CHATBOT
# ===============================
st.sidebar.title("🤖 Fintech AI Assistant")

q = st.sidebar.text_input("Ask about risk, companies, or scores")

if q:

    if AI_AVAILABLE:
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a senior credit risk analyst in a bank."},
                    {"role": "user", "content": q}
                ]
            )
            st.sidebar.write(response.choices[0].message.content)

        except Exception as e:
            st.sidebar.write("AI error:", e)

    else:
        st.sidebar.write("AI not configured. Add OPENAI_API_KEY in Streamlit secrets.")
