import streamlit as st
import pandas as pd

from src.config import (
    MODEL_PATH,
    SCALER_PATH,
    FEATURES_PATH,
    CHURN_THRESHOLD,
    SAMPLE_DATA_PATH
)
from src.inference import load_resources, predict_churn
from src.data_processing import (
    get_template_data,
    preprocess_data,
    check_missing_columns,
    load_sample_data
)
from src.visualizations import (
    style_dataframe,
    plot_density_chart,
    plot_interactive_histogram
)

def setup_page():
    st.set_page_config(
        page_title="Customer Churn Prediction",
        layout="wide",
        initial_sidebar_state="expanded"
    )

def render_header():
    st.title("Customer Churn Prediction Dashboard")
    st.markdown(
        "Identify high-risk customers, analyze churn probability distributions, "
        "and explore actionable retention insights using XGBoost."
    )
    st.divider()

def render_sidebar(features):
    with st.sidebar:
        st.header("Data Management")
        st.markdown("Upload customer data or explore the pre-loaded sample dataset.")

        uploaded_file = st.file_uploader("Upload Customer Data (CSV)", type=["csv"])
        use_sample = st.checkbox(
            "Use Sample Dataset",
            value=(uploaded_file is None),
            help="Load pre-built customer test data"
        )

        st.divider()

        st.subheader("Data Template")
        template_df = get_template_data(features)
        csv_data = template_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download CSV Template",
            data=csv_data,
            file_name="customer_template.csv",
            mime="text/csv"
        )

    return uploaded_file, use_sample

def render_predictions(df_results):
    st.subheader("Overview Metrics")
    total_customers = len(df_results)
    high_risk = int(df_results["High Risk"].sum())
    avg_probability = float(df_results["Churn Probability"].mean())

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(label="Total Analyzed Customers", value=total_customers)
    with col2:
        st.metric(
            label="High Risk Customers",
            value=high_risk,
            help=f"Customers with Churn Probability > {int(CHURN_THRESHOLD * 100)}%"
        )
    with col3:
        st.metric(label="Average Churn Probability", value=f"{avg_probability:.1%}")

    st.divider()

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
            st.plotly_chart(fig_density, use_container_width=True)

        with col_chart2:
            fig_interactive = plot_interactive_histogram(df_results, CHURN_THRESHOLD)
            st.plotly_chart(fig_interactive, use_container_width=True)

def main():
    setup_page()

    model, scaler, features = load_resources(MODEL_PATH, SCALER_PATH, FEATURES_PATH)
    if model is None or scaler is None or features is None:
        st.error(
            "Failed to load model resources. Please ensure '.pkl' files are available in the 'models/' directory."
        )
        st.stop()

    render_header()
    uploaded_file, use_sample = render_sidebar(features)

    raw_df = None
    if uploaded_file is not None:
        raw_df = pd.read_csv(uploaded_file)
    elif use_sample:
        raw_df = load_sample_data(SAMPLE_DATA_PATH)
        if raw_df is None:
            raw_df = get_template_data(features)

    if raw_df is not None:
        df_clean = preprocess_data(raw_df)
        missing_cols = check_missing_columns(df_clean, features)

        if missing_cols:
            st.error(f"Missing columns in uploaded data: {', '.join(missing_cols)}")
        else:
            churn_proba = predict_churn(model, scaler, df_clean, features)

            df_results = raw_df.copy()
            df_results["Churn Probability"] = churn_proba
            df_results["High Risk"] = (df_results["Churn Probability"] > CHURN_THRESHOLD).astype(int)

            render_predictions(df_results)
    else:
        st.info("Awaiting data. Please upload a customer datasheet or toggle 'Use Sample Dataset' in the sidebar.")

if __name__ == "__main__":
    main()
