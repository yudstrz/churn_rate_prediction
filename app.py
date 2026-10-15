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
    plot_interactive_histogram,
    plot_feature_importance,
    plot_value_risk_scatter
)

def inject_custom_styles():
    st.markdown(
        """
        <style>
        /* Base typography & layout */
        html, body, [class*="css"] {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
            color: #F8FAFC;
        }

        /* Prevent horizontal overflow on mobile */
        [data-testid="stAppViewContainer"] {
            overflow-x: hidden;
            padding-bottom: 2rem;
        }

        /* Responsive Metric Cards */
        [data-testid="stMetric"] {
            background-color: #1E293B;
            border: 1px solid #334155;
            border-radius: 8px;
            padding: 16px;
            box-shadow: none;
            transition: border-color 0.2s ease;
        }

        [data-testid="stMetric"]:hover {
            border-color: #475569;
        }

        [data-testid="stMetricLabel"] {
            color: #94A3B8 !important;
            font-size: 0.85rem !important;
            font-weight: 500;
        }

        [data-testid="stMetricValue"] {
            color: #F8FAFC !important;
            font-size: 1.6rem !important;
            font-weight: 600;
        }

        /* Mobile reflow: Stack metric columns on screens <= 768px */
        @media (max-width: 768px) {
            [data-testid="column"] {
                width: 100% !important;
                flex: 1 1 100% !important;
                min-width: 100% !important;
                margin-bottom: 12px;
            }

            .main .block-container {
                padding-left: 1rem !important;
                padding-right: 1rem !important;
                padding-top: 1.5rem !important;
            }

            h1 {
                font-size: 1.65rem !important;
            }

            h2 {
                font-size: 1.35rem !important;
            }

            h3 {
                font-size: 1.15rem !important;
            }
        }

        /* Accessible 44px minimum tap targets */
        button, [role="tab"], .stDownloadButton > button, .stCheckbox > label {
            min-height: 44px !important;
            display: inline-flex !important;
            align-items: center !important;
        }

        /* Minimalist Tabs */
        button[role="tab"] {
            padding: 10px 18px !important;
            font-size: 0.95rem !important;
            font-weight: 500 !important;
            color: #94A3B8 !important;
            border-radius: 6px 6px 0 0 !important;
        }

        button[role="tab"][aria-selected="true"] {
            color: #38BDF8 !important;
            border-bottom: 2px solid #38BDF8 !important;
        }

        /* Form elements & sliders */
        .stSelectbox, .stSlider {
            margin-bottom: 1rem;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

def setup_page():
    st.set_page_config(
        page_title="Customer Churn Prediction",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    inject_custom_styles()

def render_header():
    st.title("Customer Churn Prediction Dashboard")
    st.markdown(
        "Identify high-risk accounts, examine revenue exposure, "
        "and simulate retention outcomes with XGBoost."
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
            mime="text/csv",
            use_container_width=True
        )

    return uploaded_file, use_sample

def render_predictions(df_results, model, scaler, features):
    charge_cols = [c for c in ["total day charge", "total eve charge", "total night charge", "total intl charge"] if c in df_results.columns]
    total_customers = len(df_results)
    high_risk = int(df_results["High Risk"].sum())
    avg_probability = float(df_results["Churn Probability"].mean())
    risk_revenue = float(df_results.loc[df_results["High Risk"] == 1, charge_cols].sum().sum()) if charge_cols else 0.0

    st.subheader("Overview Metrics")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(label="Total Customers", value=total_customers)
    with col2:
        st.metric(label="High Risk Count", value=high_risk, help=f"Probability > {int(CHURN_THRESHOLD * 100)}%")
    with col3:
        st.metric(label="Avg Churn Probability", value=f"{avg_probability:.1%}")
    with col4:
        st.metric(label="Revenue at Risk ($)", value=f"${risk_revenue:,.2f}", help="Total monthly charges from high-risk accounts")

    st.divider()

    tab1, tab2, tab3, tab4 = st.tabs([
        "Data & Predictions",
        "Risk & Revenue Analytics",
        "Model Drivers (XAI)",
        "What-If Simulator"
    ])

    with tab1:
        st.subheader("Customer Prediction Data")
        styled_df = style_dataframe(df_results)
        st.dataframe(styled_df, use_container_width=True)

    with tab2:
        st.subheader("Risk Distribution & Revenue Exposure")
        col_c1, col_c2 = st.columns(2)
        with col_c1:
            st.plotly_chart(plot_density_chart(df_results, CHURN_THRESHOLD), use_container_width=True)
        with col_c2:
            st.plotly_chart(plot_interactive_histogram(df_results, CHURN_THRESHOLD), use_container_width=True)

        st.plotly_chart(plot_value_risk_scatter(df_results, CHURN_THRESHOLD), use_container_width=True)

    with tab3:
        st.subheader("Explainable AI: Key Feature Drivers")
        st.markdown("Global feature weights determined by the trained XGBoost model.")
        fi_chart = plot_feature_importance(model, features)
        if fi_chart is not None:
            st.plotly_chart(fi_chart, use_container_width=True)
        else:
            st.info("Feature importance data is unavailable for the current model.")

    with tab4:
        st.subheader("What-If Counterfactual Simulator")
        st.markdown("Select a customer and adjust operational levers to test retention outcomes.")

        indices = df_results.index.tolist()
        high_risk_indices = df_results[df_results["High Risk"] == 1].index.tolist()
        default_idx = high_risk_indices[0] if high_risk_indices else indices[0]

        selected_idx = st.selectbox("Select Customer Record Index:", indices, index=indices.index(default_idx))
        target_row = df_results.loc[selected_idx].copy()
        base_prob = float(target_row["Churn Probability"])

        sim_col1, sim_col2 = st.columns([1, 1])
        with sim_col1:
            st.markdown("**Simulate Levers**")
            sim_cs = st.slider("Customer Service Calls", 0, 10, int(target_row.get("customer service calls", 1)))
            sim_day_mins = st.slider("Total Day Minutes", 0.0, 400.0, float(target_row.get("total day minutes", 150.0)), step=5.0)
            cur_intl = str(target_row.get("international plan", "no")).lower()
            sim_intl = st.selectbox("International Plan", ["no", "yes"], index=1 if cur_intl in ["yes", "1"] else 0)

        sim_df = pd.DataFrame([target_row])
        sim_df["customer service calls"] = sim_cs
        sim_df["total day minutes"] = sim_day_mins
        sim_df["total day charge"] = sim_day_mins * 0.17
        sim_df["international plan"] = sim_intl

        sim_clean = preprocess_data(sim_df)
        sim_prob = float(predict_churn(model, scaler, sim_clean, features)[0])

        with sim_col2:
            st.markdown("**Predicted Impact**")
            st.metric(
                label="Simulated Churn Probability",
                value=f"{sim_prob:.1%}",
                delta=f"{(sim_prob - base_prob):.1%}",
                delta_color="inverse"
            )
            if sim_prob > CHURN_THRESHOLD:
                st.error(f"High Risk (> {int(CHURN_THRESHOLD*100)}%)")
            else:
                st.success(f"Retained / Safe (<= {int(CHURN_THRESHOLD*100)}%)")

def main():
    setup_page()

    model, scaler, features = load_resources(MODEL_PATH, SCALER_PATH, FEATURES_PATH)
    if model is None or scaler is None or features is None:
        st.error("Failed to load model resources from the 'models/' directory.")
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

            render_predictions(df_results, model, scaler, features)
    else:
        st.info("Awaiting data. Please upload a customer datasheet or toggle 'Use Sample Dataset' in the sidebar.")

if __name__ == "__main__":
    main()
