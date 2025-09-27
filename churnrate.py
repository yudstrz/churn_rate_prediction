# ============================================
# Streamlit App - Customer Churn Prediction
# ============================================
# Installation (if needed):
# pip install -r requirements.txt

import streamlit as st
import pandas as pd
import joblib
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px

# ================================
# Load Saved Model, Scaler, and Features
# ================================
model = joblib.load("xgb_churn_model.pkl")
scaler = joblib.load("scaler.pkl")
features = joblib.load("feature_names.pkl")

# ================================
# App Title
# ================================
st.title("Customer Churn Prediction")

# ================================
# Download Template CSV
# ================================
st.subheader("📥 Download Template Data")

if st.button("Generate Template CSV"):
    # Dummy dataset for template
    dummy_data = {
        "account length": [107, 137, 84, 75, 118],
        "area code": [415, 415, 408, 510, 415],
        "international plan": ["no", "no", "yes", "no", "no"],   # categorical
        "voice mail plan": ["yes", "no", "no", "yes", "no"],      # categorical
        "number vmail messages": [26, 0, 0, 37, 0],
        "total day minutes": [161.6, 243.4, 299.4, 166.7, 223.4],
        "total day calls": [123, 114, 71, 113, 98],
        "total day charge": [27.47, 41.38, 50.90, 28.34, 37.98],
        "total eve minutes": [195.5, 121.2, 61.9, 148.3, 220.6],
        "total eve calls": [103, 110, 88, 122, 101],
        "total eve charge": [16.62, 10.30, 5.26, 12.61, 18.75],
        "total night minutes": [254.4, 162.6, 196.9, 186.9, 203.9],
        "total night calls": [103, 104, 89, 121, 118],
        "total night charge": [11.45, 7.32, 8.86, 8.41, 9.18],
        "total intl minutes": [13.7, 12.2, 6.6, 10.1, 11.2],
        "total intl calls": [3, 5, 7, 3, 5],
        "total intl charge": [3.70, 3.29, 1.78, 2.73, 3.02],
        "customer service calls": [1, 1, 0, 2, 3]
    }

    # Keep only features used by model
    dummy_data = {k: v for k, v in dummy_data.items() if k in features}
    template = pd.DataFrame(dummy_data)

    # Convert to CSV and allow download
    csv = template.to_csv(index=False).encode()
    st.download_button(
        label="⬇️ Download CSV Template",
        data=csv,
        file_name="customer_template.csv",
        mime="text/csv"
    )

# ================================
# Upload Customer Data
# ================================
st.subheader("📤 Upload Data for Prediction")

uploaded = st.file_uploader("Upload customer data (CSV)", type=["csv"])

if uploaded:
    df_new = pd.read_csv(uploaded)

    # Convert categorical plans into binary (0/1)
    if "international plan" in df_new.columns:
        df_new["international plan"] = (
            df_new["international plan"]
            .astype(str)
            .str.lower()
            .map({"yes": 1, "no": 0})
        )
    if "voice mail plan" in df_new.columns:
        df_new["voice mail plan"] = (
            df_new["voice mail plan"]
            .astype(str)
            .str.lower()
            .map({"yes": 1, "no": 0})
        )

    # Check for missing columns
    missing_cols = set(features) - set(df_new.columns)
    if missing_cols:
        st.error(f"⚠️ Missing columns: {missing_cols}")
    else:
        # Keep only model features
        X_new = df_new[features].copy()

        # Drop state column if present
        if "state" in X_new.columns:
            X_new = X_new.drop(columns=["state"])

        # Scale features
        X_scaled = scaler.transform(X_new)

        # Predict churn probabilities
        churn_proba = model.predict_proba(X_scaled)[:, 1]
        df_new["Churn Probability"] = churn_proba
        df_new["High Risk"] = (df_new["Churn Probability"] > 0.7).astype(int)

        # Show prediction results
        st.subheader("🔎 Prediction Results")
        st.dataframe(df_new)

        # High-risk summary
        high_risk = df_new["High Risk"].sum()
        st.write(f"⚠️ Total High Risk Customers (>70% prob): **{high_risk}**")

        # ================================
        # Distribution Plot of Churn Probability
        # ================================
        st.subheader("📊 Churn Probability Distribution")

        # Seaborn + Matplotlib
        fig, ax = plt.subplots()
        sns.histplot(df_new["Churn Probability"], bins=20, kde=True, ax=ax, color="skyblue")
        ax.axvline(0.7, color="red", linestyle="--", label="High Risk Threshold (0.7)")
        ax.set_title("Churn Probability Distribution")
        ax.set_xlabel("Probability")
        ax.set_ylabel("Count")
        ax.legend()
        st.pyplot(fig)

        # Optional: Plotly interactive chart
        fig_plotly = px.histogram(
            df_new,
            x="Churn Probability",
            nbins=20,
            title="Churn Probability Distribution (Interactive)",
            color_discrete_sequence=["#636EFA"]
        )
        fig_plotly.add_vline(x=0.7, line_dash="dash", line_color="red")
        st.plotly_chart(fig_plotly, use_container_width=True)
