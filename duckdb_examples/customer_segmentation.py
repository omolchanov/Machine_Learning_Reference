import duckdb
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans

import matplotlib.pyplot as plt
import seaborn as sns

# Configuration
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', 1)


if __name__ == '__main__':

    # Generate sample data
    np.random.seed(42)
    n_customers = 100
    n_orders = 500

    orders = pd.DataFrame({
        'order_id': range(n_orders),
        'customer_id': np.random.choice(range(1, n_customers + 1), size=n_orders),
        'amount': np.random.exponential(scale=100, size=n_orders).round(2),
        'date': pd.date_range('2024-01-01', periods=n_orders, freq='D')
    })

    # Use DuckDB to aggregate customer behavior
    con = duckdb.connect()
    con.register('orders', orders)

    sql = """
    SELECT 
        customer_id,
        COUNT(*) AS num_orders,
        SUM(amount) AS total_spent,
        AVG(amount) AS avg_order_value,
        MAX(date) AS last_purchase_date
    FROM orders
    GROUP BY customer_id
    ORDER BY total_spent DESC
    """

    customer_features = con.execute(sql).fetchdf()
    print(customer_features.shape)
    print(customer_features.head(10))

    # KMeans Clustering (based on frequency + total spend)
    X = customer_features[['num_orders', 'total_spent']]

    kmeans = KMeans(n_clusters=3, random_state=42)
    customer_features['cluster'] = kmeans.fit_predict(X)

    # Step 4: Visualize clusters
    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        data=customer_features,
        x='num_orders',
        y='total_spent',
        hue='cluster',
        palette='Set2',
        s=100
    )
    plt.title("Customer Segments by Orders and Spend")
    plt.xlabel("Number of Orders")
    plt.ylabel("Total Spent ($)")
    plt.legend(title="Cluster")
    plt.tight_layout()
    plt.show()
