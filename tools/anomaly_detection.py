import pandas as pd
from sklearn.ensemble import IsolationForest
from typing import Tuple

def detect_anomalies(df: pd.DataFrame, contamination: float = 0.1) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Detect anomalies in the salary columns using Isolation Forest.

    Parameters:
    - df (pd.DataFrame): DataFrame containing 'Low salary' and 'High salary' columns.
    - contamination (float): The proportion of outliers in the data set.

    Returns:
    - cleaned_df (pd.DataFrame): DataFrame without anomalies.
    - anomalies_df (pd.DataFrame): DataFrame containing only the anomalies.
    """
    required_columns = ['Low salary', 'High salary']

    # Verify the necessary columns exist and are numeric
    for col in required_columns:
        if col not in df.columns:
            raise KeyError(f"DataFrame must contain '{col}' column")
        if not pd.api.types.is_numeric_dtype(df[col]):
            raise TypeError(f"'{col}' column must contain numeric data")

    # Detect anomalies using Isolation Forest
    isolation_forest_model = IsolationForest(contamination=contamination, random_state=42)
    
    # Fit the model and predict anomalies
    df['anomaly'] = isolation_forest_model.fit_predict(df[required_columns])

    # Separate the data into 'clean' and 'anomalies'
    cleaned_df = df[df['anomaly'] == 1].drop(columns=['anomaly'])
    anomalies_df = df[df['anomaly'] == -1].drop(columns=['anomaly'])

    return cleaned_df, anomalies_df
