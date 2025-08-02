"""
Product Recommendation
Dataset: User-product ratings
Goal: Recommend products based on user preferences
Tools: Collaborative filtering or matrix factorization
"""

import duckdb
import pandas as pd
import numpy as np

from sklearn.decomposition import TruncatedSVD

# Configuration
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', 1)


if __name__ == '__main__':

    # Generate synthetic user-product rating data
    np.random.seed(42)
    n_users = 100
    n_products = 30

    ratings = [
        (user_id, product_id, np.random.choice([1, 2, 3, 4, 5], p=[0.05, 0.1, 0.3, 0.4, 0.15]))
        for user_id in range(1, n_users + 1)
        for product_id in np.random.choice(range(1, n_products + 1), size=10, replace=False)
    ]

    df = pd.DataFrame(ratings, columns=["user_id", "product_id", "rating"])

    # Create DuckDB and load the data
    con = duckdb.connect()
    con.register("recommendations", df)
    con.execute("CREATE TABLE ratings AS SELECT * FROM recommendations")

    # DuckDB: Summary stats and filtering
    print("\nTop-rated products (avg rating, min 10 ratings):")
    summary = con.execute("""
        SELECT 
            product_id,
            COUNT(*) AS num_ratings,
            ROUND(AVG(rating), 2) AS avg_rating
        FROM ratings
        GROUP BY product_id
        HAVING num_ratings >= 10
        ORDER BY avg_rating DESC
        LIMIT 5
    """).fetchdf()
    print(summary)

    # Get distinct product IDs to build pivot query dynamically
    product_ids = con.execute("SELECT DISTINCT product_id FROM ratings ORDER BY product_id").fetchdf()[
        "product_id"].tolist()

    # Build the SQL manually
    pivot_parts = [
        f"MAX(CASE WHEN product_id = {pid} THEN rating ELSE 0 END) AS product_{pid}"
        for pid in product_ids
    ]

    pivot_sql = f"""
        SELECT user_id,
               {', '.join(pivot_parts)}
        FROM ratings
        GROUP BY user_id
        ORDER BY user_id
    """

    pivot_df = con.execute(pivot_sql).fetchdf()
    pivot_df = pivot_df.set_index("user_id")

    # Matrix Factorization (SVD)
    # Truncated Singular Value Decomposition. It performs low-rank approximation of a matrix —
    # extracting its most important structure with fewer components.
    svd = TruncatedSVD(n_components=15, random_state=42)
    user_features = svd.fit_transform(pivot_df)
    item_features = svd.components_

    # Recommend top products for a user
    def recommend(user_id: int, N: int = 5):
        user_idx = user_id - 1
        already_rated = pivot_df.columns[pivot_df.iloc[user_idx] > 0]

        preds = predicted[user_idx]
        recs = [(int(col.replace("product_", "")), preds[i])
                for i, col in enumerate(pivot_df.columns) if col not in already_rated]
        top = sorted(recs, key=lambda x: -x[1])[:N]
        return top


    predicted = np.dot(user_features, item_features)

    user_id = 10
    top_recs = recommend(user_id)

    print(f"\n▶️ Top recommendations for user {user_id}:")
    for pid, score in top_recs:
        print(f"Product {pid} — predicted rating: {score:.2f}")
