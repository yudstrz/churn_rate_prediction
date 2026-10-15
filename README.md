# Customer Churn Prediction Dashboard

A machine learning interactive dashboard for predicting customer churn. Upload customer data to identify high-risk accounts, analyze churn probability distributions, and generate actionable retention insights, built with XGBoost and Streamlit.

## Features

- **Predictive Analytics:** Upload customer demographic and usage data (CSV) to get instant churn probability predictions.
- **One-Click Sample Data:** Test the dashboard immediately with the bundled realistic dataset without manual uploads.
- **High-Risk Customer Identification:** Automatically flags accounts with a >70% calculated risk of canceling their subscription.
- **Visualizations:** Probability density distribution using Matplotlib and Seaborn, plus interactive histograms built with Plotly.
- **Summary Metrics:** Overview of the total analyzed base, high-risk volume, and average churn probability.
- **Downloadable CSV Template:** Download the expected schema template directly from the app.

## Tech Stack

- **Dashboard Framework:** Streamlit
- **Machine Learning / Inference:** XGBoost, scikit-learn, joblib
- **Data Manipulation:** pandas, numpy
- **Visualization:** matplotlib, seaborn, plotly
- **Quality Assurance:** Python standard library `unittest`

## Project Structure

```text
churn_rate_prediction/
├── .streamlit/
│   └── config.toml             # Streamlit UI theme and server configuration
├── data/                       # Datasets and sample templates
│   ├── README.md               # Data dictionary and feature documentation
│   └── sample_customers.csv    # Ready-to-use sample dataset for testing
├── models/                     # Pre-trained machine learning artifacts
│   ├── feature_names.pkl       # Feature list and ordering expected by model
│   ├── scaler.pkl              # Fitted scaler
│   └── xgb_churn_model.pkl     # Trained XGBoost classification model
├── src/                        # Application backend
│   ├── __init__.py             # Package declaration
│   ├── config.py               # Path resolutions and thresholds
│   ├── data_processing.py      # Preprocessing, data cleaning, and template generation
│   ├── inference.py            # Model loading and batch inference
│   └── visualizations.py       # Charting routines
├── tests/                      # Automated unit test suite
│   ├── __init__.py
│   ├── test_config.py          # Configuration and file path validation tests
│   └── test_data_processing.py # Data preprocessing and column check tests
├── .env.example                # Environment variables template
├── .gitignore                  # Git ignore rules for Python and Streamlit
├── app.py                      # Primary Streamlit application entry point
├── churnrate.py                # Legacy entry point (backward compatibility)
├── requirements.txt            # Dependencies
├── LICENSE                     # Open-source license
└── README.md                   # Project documentation
```

## Setup and Installation

### 1. Clone the Repository

```bash
git clone https://github.com/yudstrz/churn_rate_prediction.git
cd churn_rate_prediction
```

### 2. Create and Activate Virtual Environment

**On Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**On macOS/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

## Running the Application

Launch the Streamlit web dashboard:

```bash
streamlit run app.py
```

> Note: The legacy entry command `streamlit run churnrate.py` remains supported and forwards to `app.py`.

## Running Automated Tests

Run the built-in test suite:

```bash
python -m unittest discover tests
```

## Usage Instructions

1. **Launch the dashboard:** Follow the run command above to open the application in your browser (`http://localhost:8501`).
2. **Instant Test:** By default, the app loads `data/sample_customers.csv` via the **"Use Sample Dataset"** toggle so you can immediately see predictions and analytics.
3. **Upload Your Data:** Uncheck the sample toggle or use the **File Uploader** in the left sidebar to upload your own customer CSV.
4. **Download Template:** Click **"Download CSV Template"** in the sidebar if you need an empty reference format matching the model's schema.
5. **Explore Insights:**
   - Review **Overview Metrics** (Total Analyzed, High Risk Volume, Average Churn Probability).
   - Navigate the **"Data & Predictions"** tab to inspect row-level predictions.
   - Navigate the **"Probability Distribution"** tab for visual risk curves and histograms.
