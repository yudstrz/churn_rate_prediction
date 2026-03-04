import pandas as pd

def get_template_data(features):
    """Generate dummy data for the CSV download template."""
    dummy_data = {
        "account length": [107, 137, 84, 75, 118],
        "area code": [415, 415, 408, 510, 415],
        "international plan": ["no", "no", "yes", "no", "no"],
        "voice mail plan": ["yes", "no", "no", "yes", "no"],
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
