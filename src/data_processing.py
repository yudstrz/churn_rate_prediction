import numpy as pd # Using pd alias for backwards compat with `pd.DataFrame` downstream. Note: Using numpy for random generation, then wrapping in pandas
import numpy as np

def get_template_data(features):
    """Generate 100 rows of dummy data for the CSV download template, with realistic high/low risk distributions."""
    np.random.seed(42)  # For reproducibility
    n_samples = 100
    
    # Generate mix of normal customers and "high risk" profiles
    # High risk typically has: more customer service calls, high day charges, international plan = yes
    is_high_risk = np.random.choice([0, 1], size=n_samples, p=[0.75, 0.25])
    
    dummy_data = {
        "account length": np.random.randint(1, 243, n_samples),
        "area code": np.random.choice([408, 415, 510], n_samples),
        "international plan": np.where(is_high_risk, np.random.choice(["yes", "no"], n_samples, p=[0.6, 0.4]), np.random.choice(["yes", "no"], n_samples, p=[0.05, 0.95])),
        "voice mail plan": np.random.choice(["yes", "no"], n_samples, p=[0.3, 0.7]),
        "number vmail messages": np.random.randint(0, 50, n_samples),
        "total day minutes": np.where(is_high_risk, np.random.normal(250, 40, n_samples), np.random.normal(150, 50, n_samples)),
        "total day calls": np.random.randint(50, 150, n_samples),
    }
    
    # Derive charges based on minutes (approximate rates)
    dummy_data["total day charge"] = dummy_data["total day minutes"] * 0.17
    
    dummy_data.update({
        "total eve minutes": np.random.normal(200, 50, n_samples),
        "total eve calls": np.random.randint(50, 150, n_samples),
    })
    dummy_data["total eve charge"] = dummy_data["total eve minutes"] * 0.085
    
    dummy_data.update({
        "total night minutes": np.random.normal(200, 50, n_samples),
        "total night calls": np.random.randint(50, 150, n_samples),
    })
    dummy_data["total night charge"] = dummy_data["total night minutes"] * 0.045
    
    dummy_data.update({
        "total intl minutes": np.random.normal(10, 3, n_samples),
        "total intl calls": np.random.randint(1, 10, n_samples),
    })
    dummy_data["total intl charge"] = dummy_data["total intl minutes"] * 0.27
    
    # High risk correlates strongly with customer service calls
    dummy_data["customer service calls"] = np.where(is_high_risk, np.random.randint(3, 8, n_samples), np.random.randint(0, 3, n_samples))
    
    # Ensure all values are non-negative
    for col in dummy_data:
        if isinstance(dummy_data[col][0], (int, float)):
             dummy_data[col] = np.maximum(0, dummy_data[col])

    if features is not None:
        dummy_data_filtered = {k: v for k, v in dummy_data.items() if k in features}
        return pd.DataFrame(dummy_data_filtered)
    return pd.DataFrame(dummy_data)

def preprocess_data(df):
    """Clean and preprocess uploaded data for modeling."""
    df_clean = df.copy()
    
    # Convert categorical plans into binary (0/1)
    if "international plan" in df_clean.columns:
        df_clean["international plan"] = (
            df_clean["international plan"]
            .astype(str)
            .str.lower()
            .map({"yes": 1, "no": 0})
        )
    if "voice mail plan" in df_clean.columns:
        df_clean["voice mail plan"] = (
            df_clean["voice mail plan"]
            .astype(str)
            .str.lower()
            .map({"yes": 1, "no": 0})
        )
    
    # Drop state column if present
    if "state" in df_clean.columns:
        df_clean = df_clean.drop(columns=["state"])
        
    return df_clean

def check_missing_columns(df, features):
    """Check if uploaded dataframe contains all the required model features."""
    return set(features) - set(df.columns)
