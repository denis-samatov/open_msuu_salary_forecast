import pandas as pd
from sklearn.ensemble import IsolationForest
from typing import Tuple


def detect_anomalies(df: pd.DataFrame, contamination: float = 0.1, n_estimators: int = 100, max_samples: float = 'auto', n_jobs: int = -1) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Detect anomalies in the salary columns using Isolation Forest.

    Parameters:
    - df (pd.DataFrame): DataFrame containing 'Low salary' and 'High salary' columns.
    - contamination (float): The proportion of outliers in the dataset.
    - n_estimators (int): The number of base estimators in the ensemble.
    - max_samples (float or int): The number of samples to draw to train each base estimator.
    - n_jobs (int): The number of jobs to run in parallel. -1 means using all processors.

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
    isolation_forest_model = IsolationForest(
        contamination=contamination, 
        n_estimators=n_estimators, 
        max_samples=max_samples, 
        random_state=42,
        n_jobs=n_jobs  # Utilize all available processors
    )
    
    # Fit the model and predict anomalies
    df['anomaly'] = isolation_forest_model.fit_predict(df[required_columns])

    # Separate the data into 'clean' and 'anomalies'
    cleaned_df = df[df['anomaly'] == 1].drop(columns=['anomaly'])
    anomalies_df = df[df['anomaly'] == -1].drop(columns=['anomaly'])

    return cleaned_df, anomalies_df
