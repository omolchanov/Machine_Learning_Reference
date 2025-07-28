import duckdb
import pandas as pd
import numpy as np

# Configuration
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', 1)


def ingest_data():
    np.random.seed(42)  # for reproducibility

    n = 10000  # number of rows

    # Generate synthetic data
    data = {
        'id': np.arange(1, n + 1),
        'age': np.random.randint(20, 60, size=n).astype(float),
        'income': np.random.randint(30000, 120000, size=n).astype(float),
        'gender': np.random.choice(['M', 'F'], size=n)
    }

    df = pd.DataFrame(data)

    # Randomly introduce missing values (~10-20% missing per column)
    def inject_missing(df, column, frac=0.15):
        n_missing = int(frac * len(df))
        missing_indices = np.random.choice(df.index, n_missing, replace=False)
        df.loc[missing_indices, column] = np.nan

    inject_missing(df, 'age', frac=0.15)
    inject_missing(df, 'income', frac=0.1)
    inject_missing(df, 'gender', frac=0.2)

    return df


if __name__ == '__main__':
    df = ingest_data()

    # Connect to duckdb
    con = duckdb.connect()
    con.register('people', df)

    # 1. Find missing counts and % missing per column using SQL aggregates
    sql = """
    SELECT 
        'age' AS column_name,
        COUNT(*) AS total_rows,
        COUNT(age) AS non_missing,
        COUNT(*) - COUNT(age) AS missing_count,
        ROUND(100.0 * (COUNT(*) - COUNT(age)) / COUNT(*), 2) AS missing_percent
    FROM people
    UNION ALL
    SELECT 
        'income',
        COUNT(*),
        COUNT(income),
        COUNT(*) - COUNT(income),
        ROUND(100.0 * (COUNT(*) - COUNT(income)) / COUNT(*), 2)
    FROM people
    UNION ALL
    SELECT 
        'gender',
        COUNT(*),
        COUNT(gender),
        COUNT(*) - COUNT(gender),
        ROUND(100.0 * (COUNT(*) - COUNT(gender)) / COUNT(*), 2)
    FROM people
    """
    result = con.execute(sql).fetch_df()
    print("Missing Data Summary:")
    print(result)

    # Basic Descriptive Stats Ignoring Nulls
    sql = """
    SELECT 
        AVG(age) AS avg_age,
        MIN(age) AS min_age,
        MAX(age) AS max_age,
        AVG(income) AS avg_income
    FROM people
    """
    result = con.execute(sql).fetch_df()
    print("\nDescription Summary:")
    print(result)

    # Grouped Missing Value Analysis
    sql = """
    SELECT gender, COUNT(*) AS total, 
       SUM(CASE WHEN income IS NULL THEN 1 ELSE 0 END) AS missing_income
    FROM df
    GROUP BY gender
    """
    result = con.execute(sql).fetch_df()
    print("\nGrouped Missing Value Analysis:")
    print(result)

    # Distribution Buckets with Missing Info
    sql = """
    SELECT 
        CASE 
            WHEN age < 30 THEN '20s'
            WHEN age < 40 THEN '30s'
            WHEN age < 50 THEN '40s'
            ELSE '50+'
        END AS age_group,
        COUNT(*) AS total_in_group,
        SUM(CASE WHEN income IS NULL THEN 1 ELSE 0 END) AS missing_income_in_group
    FROM df
    WHERE age IS NOT NULL
    GROUP BY age_group
    ORDER BY age_group
    """
    result = con.execute(sql).fetch_df()
    print("\nDistribution Buckets with Missing Info:")
    print(result)

    con.close()
