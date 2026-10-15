import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"
MODEL_PATH = str(MODELS_DIR / "xgb_churn_model.pkl")
SCALER_PATH = str(MODELS_DIR / "scaler.pkl")
FEATURES_PATH = str(MODELS_DIR / "feature_names.pkl")
SAMPLE_DATA_PATH = str(BASE_DIR / "data" / "sample_customers.csv")
CHURN_THRESHOLD = float(os.getenv("CHURN_THRESHOLD", 0.7))
