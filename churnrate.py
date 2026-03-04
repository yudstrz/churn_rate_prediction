# ============================================
# Streamlit App - Customer Churn Prediction
# ============================================

import streamlit as st
import pandas as pd

from src.config import MODEL_PATH, SCALER_PATH, FEATURES_PATH, CHURN_THRESHOLD
from src.inference import load_resources, predict_churn
from src.data_processing import get_template_data, preprocess_data, check_missing_columns
from src.visualizations import style_dataframe, plot_density_chart, plot_interactive_histogram

# Set page configuration for a more professional look
st.set_page_config(
    page_title="Customer Churn Prediction",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ================================
# Load Resources
# ================================
model, scaler, features = load_resources(MODEL_PATH, SCALER_PATH, FEATURES_PATH)

if model is None:
    st.error("Failed to load model resources. Please ensure '.pkl' files are available in the 'models/' directory.")
    st.stop()

# ================================
# App Header
# ================================
st.title("Customer Churn Prediction")
st.markdown("Identify high-risk customers, analyze churn probability distributions, and explore individual customer profiles.")
st.divider()

# ================================
# Sidebar - Data Management
# ================================
with st.sidebar:
    st.header("Data Management")
    st.markdown("Upload customer data to generate predictions or download the template layout.")
    
    # Upload Data
    uploaded = st.file_uploader("Upload Customer Data (CSV)", type=["csv"])
    
    st.divider()
    
    # Download Template
    st.subheader("Data Template")
    template_df = get_template_data(features)
    csv = template_df.to_csv(index=False).encode()
    st.download_button(
        label="Download CSV Template",
        data=csv,
        file_name="customer_template.csv",
        mime="text/csv"
    )

# ================================
# Main Content - Predictions
# ================================
if uploaded:
    raw_df = pd.read_csv(uploaded)
    df_clean = preprocess_data(raw_df)

    # Check for missing columns
    missing_cols = check_missing_columns(df_clean, features)
    
    if missing_cols:
        st.error(f"Missing columns in uploaded data: {', '.join(missing_cols)}")
    else:
        # Predict
        churn_proba = predict_churn(model, scaler, df_clean, features)
        
        # Add predictions to dataframe
        df_results = raw_df.copy() # Keeping original format for display
        df_results["Churn Probability"] = churn_proba
        df_results["High Risk"] = (df_results["Churn Probability"] > CHURN_THRESHOLD).astype(int)

        # Summary Metrics
        st.subheader("Overview Metrics")
        total_customers = len(df_results)
        high_risk = df_results["High Risk"].sum()
        avg_probability = df_results["Churn Probability"].mean()

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(label="Total Analyzed Customers", value=total_customers)
        with col2:
            st.metric(label="High Risk Customers", value=high_risk, help=f"Probability > {CHURN_THRESHOLD*100}%")
        with col3:
            st.metric(label="Average Churn Probability", value=f"{avg_probability:.1%}")

        st.divider()

        # Tabs for Dashboard Data Organization
        tab1, tab2 = st.tabs(["Data & Predictions", "Probability Distribution"])

        with tab1:
            st.subheader("Customer Prediction Data")
            styled_df = style_dataframe(df_results)
            st.dataframe(styled_df, use_container_width=True)

        with tab2:
            st.subheader("Risk Distribution Analysis")
            col_chart1, col_chart2 = st.columns(2)
            
            with col_chart1:
                fig_density = plot_density_chart(df_results, CHURN_THRESHOLD)
                st.pyplot(fig_density)

            with col_chart2:
                fig_interactive = plot_interactive_histogram(df_results, CHURN_THRESHOLD)
                st.plotly_chart(fig_interactive, use_container_width=True)

elif not uploaded:
    st.info("Awaiting data. Please upload a customer datasheet from the sidebar to view predictions.")

