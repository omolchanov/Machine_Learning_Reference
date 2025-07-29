import duckdb
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest

import matplotlib.pyplot as plt
import seaborn as sns

# Configuration
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', 1)

if __name__ == '__main__':
    # Create synthetic sensor data
    np.random.seed(42)
    n_samples = 500

    # Normal readings
    sensor_readings = np.random.normal(loc=50, scale=5, size=n_samples)

    # Inject outliers
    outliers = np.random.uniform(low=80, high=100, size=10)
    sensor_data = np.concatenate([sensor_readings, outliers])

    df = pd.DataFrame({
        'timestamp': pd.date_range('2024-01-01', periods=len(sensor_data), freq='h'),
        'sensor_value': sensor_data
    })

    # Use DuckDB to detect outliers using IQR method
    con = duckdb.connect()

    con.register("outliers", df)
    con.execute("CREATE TABLE sensor_data AS SELECT * FROM outliers")

    sql = """
    WITH stats AS (
        SELECT 
            percentile_cont(0.25) WITHIN GROUP (ORDER BY sensor_value) AS Q1,
            percentile_cont(0.75) WITHIN GROUP (ORDER BY sensor_value) AS Q3
        FROM sensor_data
    ),
    limits AS (
        SELECT 
            Q1,
            Q3,
            (Q3 - Q1) * 1.5 AS IQR,
            Q1 - (Q3 - Q1) * 1.5 AS lower_bound,
            Q3 + (Q3 - Q1) * 1.5 AS upper_bound
        FROM stats
    )
    SELECT 
        s.*,
        CASE 
            WHEN s.sensor_value < l.lower_bound OR s.sensor_value > l.upper_bound 
            THEN TRUE ELSE FALSE 
        END AS is_outlier_iqr
    FROM sensor_data s, limits l
    """

    df = con.execute(sql).fetchdf()
    print(df[df['is_outlier_iqr'] == True])

    # Use IsolationForest for ML-based detection
    clf = IsolationForest(contamination=0.02, random_state=42)
    df['is_outlier_ml'] = clf.fit_predict(df[['sensor_value']])
    df['is_outlier_ml'] = df['is_outlier_ml'] == -1  # convert to True/False

    # Visualize
    plt.figure(figsize=(12, 6))
    sns.lineplot(x='timestamp', y='sensor_value', data=df, label='Sensor Value')
    sns.scatterplot(x='timestamp', y='sensor_value', data=df[df['is_outlier_iqr']], color='red',
                    label='IQR Outliers', s=50)
    sns.scatterplot(x='timestamp', y='sensor_value', data=df[df['is_outlier_ml']], color='purple',
                    label='ML Outliers', s=60, marker='X')
    plt.title("Sensor Data with Outliers Detected (IQR vs IsolationForest)")
    plt.xlabel("Time")
    plt.ylabel("Sensor Reading")
    plt.legend()
    plt.tight_layout()
    plt.show()
