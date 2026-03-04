# Customer Churn Prediction Dashboard

A machine learning-powered dashboard for predicting customer churn. Upload customer data to identify high-risk accounts, analyze churn probability distributions, and generate actionable retention insights — built with XGBoost and Streamlit.

## Features

- **Predictive Analytics:** Upload customer demographic and usage data (CSV) to get instant churn probability predictions.
- **High-Risk Identification:** Automatically flags customers with a >70% calculated risk of canceling their subscription.
- **Interactive Visualizations:**
  - Probability density distribution using Matplotlib & Seaborn.
  - Interactive histograms built with Plotly.
- **Summary Metrics:** Quick overview of the total analyzed base, high-risk volume, and average churn probability.
- **Modular Data Generation:** Includes a downloadable CSV template with 100 realistic customer profile examples to test the system immediately.

## Tech Stack

- **Frontend / Dashboard Framework:** [Streamlit](https://streamlit.io/)
- **Machine Learning / Inference:** `XGBoost`, `scikit-learn`
- **Data Manipulation:** `pandas`, `numpy`
- **Visualization:** `matplotlib`, `seaborn`, `plotly`

## Repository Structure

```text
churn_rate_prediction/
├── churnrate.py            # Main entry point for the Streamlit application
├── models/                 # Pre-trained machine learning artifacts
│   ├── xgb_churn_model.pkl # The trained XGBoost model
│   ├── scaler.pkl          # Feature scaler used during training
│   └── feature_names.pkl   # Expected feature list/order
├── src/                    # Application source code modules
│   ├── __init__.py
│   ├── config.py           # Universal path references and thresholds
│   ├── data_processing.py  # Data cleaning and template generation logic
│   ├── inference.py        # Scripts to load models and predict churn
│   └── visualizations.py   # Seperated charting logic (Seaborn & Plotly)
├── requirements.txt        # Deployment dependencies
└── README.md
```

## Setup and Installation

### Local Development

1. **Clone the repository:**
   ```bash
   git clone https://github.com/YourUsername/churn_rate_prediction.git
   cd churn_rate_prediction
   ```

2. **Create a virtual environment (Optional but Recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application:**
   ```bash
   streamlit run churnrate.py
   ```

## Usage Instructions

1. Upon launching the app, click **"Download CSV Template"** from the left sidebar to obtain a sample dataset with 100 realistic records.
2. In the sidebar, use the **File Uploader** to upload the template you just downloaded (or provide your own data strictly matching the template's column structure).
3. Review the overview metrics at the top to see how many customers fall under the "High Risk" category.
4. Switch between the **"Data & Predictions"** and **"Probability Distribution"** tabs to investigate individual rows or macro-trends.
