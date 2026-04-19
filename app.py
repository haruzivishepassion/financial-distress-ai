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
import os

def load_model_and_scaler():
    """Load model and scaler with multiple path attempts"""
    # Get the directory of the current script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.join(script_dir, "model")
    
    model_path = os.path.join(model_dir, "distress_model.pkl")
    scaler_path = os.path.join(model_dir, "scaler.pkl")
    
    # Alternative paths for different environments
    alternative_model_paths = [
        model_path,
        "./model/distress_model.pkl",
        "../model/distress_model.pkl",
        "/mount/src/f-d-ai2/model/distress_model.pkl"  # Streamlit Cloud path
    ]
    
    alternative_scaler_paths = [
        scaler_path,
        "./model/scaler.pkl",
        "../model/scaler.pkl",
        "/mount/src/f-d-ai2/model/scaler.pkl"  # Streamlit Cloud path
    ]
    
    # Try to load model
    model = None
    for path in alternative_model_paths:
        try:
            if os.path.exists(path):
                model = joblib.load(path)
                break
        except Exception as e:
            continue
    
    # Try to load scaler
    scaler = None
    for path in alternative_scaler_paths:
        try:
            if os.path.exists(path):
                scaler = joblib.load(path)
                break
        except Exception as e:
            continue
    
    if model is None or scaler is None:
        # Debug information
        st.error("""
        ❌ **Model Loading Error**
        Unable to load the trained model or scaler.
        """)
        st.write("Current working directory:", os.getcwd())
        st.write("Script directory:", script_dir)
        st.write("Model directory contents:", os.listdir(model_dir) if os.path.exists(model_dir) else "Model directory not found")
        st.write("""
        Please ensure:
        1. Model files exist in the 'model' directory
        2. For Streamlit Cloud: Check that model files are included in your repository
        3. For local development: Run 'train_model.py' first to generate model files
        """)
        st.stop()
    
    return model, scaler

model, scaler = load_model_and_scaler()

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
    c.execute(
        "INSERT INTO predictions (credit_score, risk) VALUES (?,?)", (score, risk)
    )
    conn.commit()


# ===============================
# FILE UPLOAD
# ===============================
file = st.file_uploader("Upload Financial Data (CSV/Excel) - Include company_name and year columns for full analysis", type=["csv", "xlsx"])

# Default to sample data if no file uploaded
company_names = None
years = None
df_original = None

# LOAD DATA (file or sample)
if file:
    if file.name.endswith(".csv"):
        df_analysis = pd.read_csv(file)
    else:
        df_analysis = pd.read_excel(file)
else:
    sample_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sample_financial_data.csv")
    if os.path.exists(sample_path):
        df_analysis = pd.read_csv(sample_path)
        st.info("📂 Using sample data (upload your own CSV to override)")
    else:
        df_analysis = None

# PROCESS DATA
df = None
if df_analysis is not None:
    # PRESERVE COMPANY NAME FOR ANALYSIS
    if "company_name" in df_analysis.columns:
        company_names = df_analysis["company_name"].copy()
    if "year" in df_analysis.columns:
        years = df_analysis["year"].copy()

    st.subheader("📂 Raw Data")
    st.dataframe(df_analysis.head())

    # STORE ORIGINAL FOR COMPANY ANALYSIS
    df_original = df_analysis.copy()

# PREPARE FOR MODEL (keep required features only)
    df = df_analysis.copy()
    years_data = None
    
    if "company_name" in df.columns:
        df = df.drop(columns=["company_name"])
    
    if "year" in df.columns:
        years_data = df["year"].copy()
        df = df.drop(columns=["year"])
    
    expected_features = ['year', 'X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X7', 'X8', 
                        'X9', 'X10', 'X11', 'X12', 'X13', 'X14', 'X15', 'X16', 'X17', 'X18']
    
    for feat in expected_features:
        if feat not in df.columns:
            df[feat] = 0
    
    df = df[expected_features]
    df = df.fillna(0)

    # Process if we have data
    if df is not None and len(df) > 0:
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
            "Filter Risk Level", df["Risk"].unique(), default=df["Risk"].unique()
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
        filtered_df["Risk"].value_counts().plot(kind="bar", ax=ax, color=['#ff6b6b', '#ffa502', '#2ed573', '#1e90ff'])
        st.pyplot(fig)

    with col2:
        st.subheader("📈 Credit Score Distribution")
        fig2, ax2 = plt.subplots()
        ax2.hist(filtered_df["Credit Score"], bins=20, color='#1e90ff', edgecolor='black')
        ax2.axvline(filtered_df["Credit Score"].mean(), color='red', linestyle='--', label='Mean')
        ax2.legend()
        st.pyplot(fig2)

    # ===============================
    # COMPANY ANALYSIS SECTION
    # ===============================
    if company_names is not None:
        st.subheader("🏢 Company Analysis")
        
        # Get unique companies
        unique_companies = company_names.unique()
        selected_company = st.selectbox("Select Company for Analysis", unique_companies)
        
        # Get company data
        company_mask = company_names == selected_company
        company_data = df_original[company_mask].copy()
        
        # Display company metrics
        col1, col2, col3, col4 = st.columns(4)
        
        avg_score = df[company_mask]["Credit Score"].mean()
        latest_score = df[company_mask]["Credit Score"].iloc[-1] if len(df[company_mask]) > 0 else 0
        risk_mode = df[company_mask]["Risk"].mode().iloc[0] if len(df[company_mask]) > 0 else "N/A"
        
        with col1:
            st.metric("Average Credit Score", f"{avg_score:.1f}")
        with col2:
            st.metric("Latest Credit Score", f"{latest_score:.1f}")
        with col3:
            st.metric("Risk Level", risk_mode)
        with col4:
            st.metric("Records", len(company_data))

        # Company trend analysis if year exists
        if years is not None:
            st.subheader(f"📈 {selected_company} - Trend Analysis")
            
            company_data_with_score = company_data.copy()
            company_data_with_score["Credit Score"] = df[company_mask]["Credit Score"].values
            company_data_with_score["Risk"] = df[company_mask]["Risk"].values
            company_data_with_score["Year"] = years[company_mask].values
            
            # Credit Score Trend
            fig_trend, ax_trend = plt.subplots(figsize=(10, 4))
            ax_trend.plot(company_data_with_score["Year"], company_data_with_score["Credit Score"], 
                         marker='o', linewidth=2, markersize=8, color='#1e90ff')
            ax_trend.set_xlabel("Year")
            ax_trend.set_ylabel("Credit Score")
            ax_trend.set_title(f"{selected_company} - Credit Score Trend")
            ax_trend.grid(True, alpha=0.3)
            st.pyplot(fig_trend)
            
            # Key Financial Metrics Trend
            st.subheader(f"📊 {selected_company} - Financial Metrics")
            numeric_cols = company_data.select_dtypes(include=[np.number]).columns.tolist()
            
            if len(numeric_cols) > 0:
                selected_metrics = st.multiselect("Select Metrics to Visualize", 
                                                   numeric_cols[:10], 
                                                   default=numeric_cols[:3] if len(numeric_cols) >= 3 else numeric_cols)
                
                if selected_metrics:
                    fig_metrics, ax_metrics = plt.subplots(figsize=(10, 4))
                    for metric in selected_metrics:
                        ax_metrics.plot(company_data_with_score["Year"], 
                                       company_data_with_score[metric], 
                                       marker='o', linewidth=2, label=metric)
                    ax_metrics.set_xlabel("Year")
                    ax_metrics.set_ylabel("Value")
                    ax_metrics.set_title(f"{selected_company} - Financial Metrics Trend")
                    ax_metrics.legend()
                    ax_metrics.grid(True, alpha=0.3)
                    st.pyplot(fig_metrics)

        # Company comparison
        st.subheader("🔍 Company Comparison")
        if len(unique_companies) > 1:
            comparison_df = pd.DataFrame({
                'Company': unique_companies,
                'Avg Credit Score': [df[company_names == c]["Credit Score"].mean() for c in unique_companies],
                'Latest Score': [df[company_names == c]["Credit Score"].iloc[-1] for c in unique_companies]
            })
            comparison_df = comparison_df.sort_values('Avg Credit Score', ascending=False)
            
            fig_comp, ax_comp = plt.subplots(figsize=(10, 4))
            x = np.arange(len(unique_companies))
            width = 0.35
            ax_comp.bar(x - width/2, comparison_df['Avg Credit Score'], width, label='Avg Score', color='#1e90ff')
            ax_comp.bar(x + width/2, comparison_df['Latest Score'], width, label='Latest Score', color='#2ed573')
            ax_comp.set_xlabel('Company')
            ax_comp.set_ylabel('Credit Score')
            ax_comp.set_title('Company Comparison - Credit Scores')
            ax_comp.set_xticks(x)
            ax_comp.set_xticklabels(comparison_df['Company'])
            ax_comp.legend()
            st.pyplot(fig_comp)
            
            st.dataframe(comparison_df)

        # Heatmap of all companies
        st.subheader("🔥 Company Health Heatmap")
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numeric_cols) > 0 and len(unique_companies) > 1:
            heatmap_data = pd.DataFrame({
                'Company': company_names,
                **{col: df[col].values for col in numeric_cols[:8]}
            }).groupby('Company').mean()
            
            fig_heat, ax_heat = plt.subplots(figsize=(12, 6))
            im = ax_heat.imshow(heatmap_data.corr(), cmap='RdYlGn', aspect='auto')
            ax_heat.set_xticks(range(len(heatmap_data.columns)))
            ax_heat.set_yticks(range(len(heatmap_data.index)))
            ax_heat.set_xticklabels(heatmap_data.columns, rotation=45, ha='right')
            ax_heat.set_yticklabels(heatmap_data.index)
            plt.colorbar(im, ax=ax_heat)
            ax_heat.set_title('Company Financial Metrics Correlation')
            st.pyplot(fig_heat)

    # ===============================
        # FEATURE IMPORTANCE
        # ===============================
        st.subheader("🧠 Key Drivers of Financial Distress")

        importance = model.feature_importances_
        features = df.drop(columns=["Credit Score", "Risk"], errors="ignore").columns[
            : len(importance)
        ]

        fig3, ax3 = plt.subplots()
        ax3.barh(features, importance)
        st.pyplot(fig3)

        # ===============================
        # INDIVIDUAL COMPANY VIEW
        # ===============================
        st.subheader("🏢 Individual Company Analysis")

        idx = st.number_input("Select Company Index", 0, len(df) - 1, 0)

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
                show=False,
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
                Credit Score: {df.iloc[i]["Credit Score"]}<br/>
                Risk: {df.iloc[i]["Risk"]}<br/>
                """
                content.append(Paragraph(text, styles["Normal"]))
                content.append(Spacer(1, 10))

            doc.build(content)

            with open("report.pdf", "rb") as f:
                st.download_button("⬇ Download Report", f, file_name="credit_report.pdf")

# ===============================
# 🤖 AI CHATBOT (ENHANCED)
# ===============================
st.sidebar.title("🤖 Fintech AI Assistant")

# Add context about uploaded data
chat_context = ""
if df is not None and company_names is not None:
    chat_context = f"""
    Current data summary:
    - Companies: {', '.join(company_names.unique())}
    - Total records: {len(df)}
    - Credit Score range: {df['Credit Score'].min():.1f} - {df['Credit Score'].max():.1f}
    - Average Credit Score: {df['Credit Score'].mean():.1f}
    """

q = st.sidebar.text_input("Ask about risk, companies, or scores", key="chat_input")

if q:
    if AI_AVAILABLE:
        try:
            # Build enhanced prompt with data context
            system_prompt = f"""You are a senior credit risk analyst in a bank. 
            You help users analyze company financial health and credit risk.
            Provide specific insights based on the data when asked about companies.
            {chat_context}"""
            
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": q},
                ],
            )
            st.sidebar.write(response.choices[0].message.content)

        except Exception as e:
            st.sidebar.write("AI error:", e)

    else:
        st.sidebar.write("AI not configured. Add OPENAI_API_KEY in Streamlit secrets.")
        # Fallback: Provide basic analysis
        if "company" in q.lower() and company_names is not None:
            st.sidebar.write("📊 **Available Companies:** " + ", ".join(company_names.unique()))
        if "score" in q.lower():
            if df is not None:
                st.sidebar.write(f"📈 **Score Range:** {df['Credit Score'].min():.1f} - {df['Credit Score'].max():.1f}")
                st.sidebar.write(f"📊 **Average Score:** {df['Credit Score'].mean():.1f}")
