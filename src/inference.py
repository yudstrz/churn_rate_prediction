import joblib
import streamlit as st

@st.cache_resource
def load_resources(model_path, scaler_path, features_path):
    """Load model, scaler, and feature names from disk."""
    try:
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        features = joblib.load(features_path)
        return model, scaler, features
    except Exception as e:
        return None, None, None

def predict_churn(model, scaler, df, features):
    """Scale data and predict churn probabilities."""
    # Keep only model features
    X_new = df[features].copy()
    
    # Scale features
    X_scaled = scaler.transform(X_new)
    
    # Predict churn probabilities
    churn_proba = model.predict_proba(X_scaled)[:, 1]
    return churn_proba
