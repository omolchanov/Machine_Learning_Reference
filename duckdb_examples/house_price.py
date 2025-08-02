"""
Predicting House Prices

Dataset: Housing data (e.g., size, location, price)
Goal: Predict price using linear regression
Tools: scikit-learn, DuckDB for feature prep
"""

import duckdb
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score, mean_absolute_percentage_error

import seaborn as sns
import matplotlib.pyplot as plt

# Configuration
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', 1)

if __name__ == '__main__':

    # Generate synthetic housing data
    np.random.seed(42)
    n = 500

    df = pd.DataFrame({
        'id': range(1, n + 1),
        'size_sqft': np.random.normal(1500, 400, n).astype(int),
        'bedrooms': np.random.randint(1, 6, n),
        'location': np.random.choice(['center', 'suburb', 'rural'], size=n, p=[0.4, 0.4, 0.2]),
    })

    # Generate price based on features (with noise)
    location_factor = {'center': 1.5, 'suburb': 1.0, 'rural': 0.7}
    df['price'] = (
        df['size_sqft'] * 150 +
        df['bedrooms'] * 10000 +
        df['location'].map(location_factor) * 50000 +
        np.random.normal(0, 20000, n)
    ).round(0)

    # Store into DuckDB
    con = duckdb.connect()
    con.register("df_houses", df)
    con.execute("CREATE TABLE houses AS SELECT * FROM df_houses")

    # Feature engineering with SQL (location one-hot encoding)
    sql = """
    SELECT
        size_sqft,
        bedrooms,
        CASE WHEN location = 'center' THEN 1 ELSE 0 END AS loc_center,
        CASE WHEN location = 'suburb' THEN 1 ELSE 0 END AS loc_suburb,
        CASE WHEN location = 'rural' THEN 1 ELSE 0 END AS loc_rural,
        price
    FROM houses
    """

    df = con.execute(sql).fetchdf()
    print(df.head())

    # Train/test split
    X = df.drop('price', axis=1)
    y = df['price']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train Linear Regression
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    mape = mean_absolute_percentage_error(y_test, y_pred) * 100
    r2 = r2_score(y_test, y_pred)

    # Visualization
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=y_test, y=y_pred)
    plt.xlabel("Actual Price")
    plt.ylabel("Predicted Price")
    plt.title("Linear Regression: Actual vs Predicted")
    plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
    plt.tight_layout()
    plt.show()

    print(f"MSE: {mae:.0f} | MAPE: {mape:.3f} | R2: {r2:.3f}")
