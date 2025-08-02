"""
User Activity Funnel
Dataset: Logs of user actions
Goal: Understand where users drop off in a process (signup → checkout)
Tools: SQL window functions, funnel plots
"""

import duckdb
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

# Configuration
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', 1)


if __name__ == '__main__':
    # Create synthetic user activity log

    np.random.seed(42)
    n_users = 300

    user_ids = np.arange(1, n_users + 1)

    # Funnel stages
    funnel_stages = ['signup', 'verify_email', 'browse', 'add_to_cart', 'checkout']

    # Simulate user drop-offs by stage
    user_events = []

    for user_id in user_ids:
        n_stages = np.random.choice(len(funnel_stages) + 1, p=[0.2, 0.25, 0.2, 0.2, 0.1, 0.05])  # Some drop early
        stages_completed = funnel_stages[:n_stages]
        for i, stage in enumerate(stages_completed):
            user_events.append({
                'user_id': user_id,
                'timestamp': pd.Timestamp('2024-01-01') + pd.Timedelta(minutes=np.random.randint(0, 10000)),
                'event': stage
            })

    df = pd.DataFrame(user_events)

    # Connect to DuckDB and store logs
    con = duckdb.connect()

    con.register("user_logs_df", df)
    con.execute("CREATE TABLE user_logs AS SELECT * FROM user_logs_df")

    # Funnel analysis using DuckDB SQL
    sql = """
    WITH distinct_events AS (
        SELECT DISTINCT user_id, event
        FROM user_logs
    ),
    funnel_counts AS (
        SELECT
            event,
            COUNT(DISTINCT user_id) AS users_at_stage
        FROM distinct_events
        GROUP BY event
    ),
    ordered_funnel AS (
        SELECT * FROM funnel_counts
        WHERE event IN ('signup', 'verify_email', 'browse', 'add_to_cart', 'checkout')
    )
    SELECT *
    FROM ordered_funnel
    ORDER BY array_position(['signup', 'verify_email', 'browse', 'add_to_cart', 'checkout'], event)
    """

    df = con.execute(sql).fetchdf()
    print(df)

    con.close()

    # Funnel plot
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df, x='event', y='users_at_stage', color='skyblue')

    plt.title("User Funnel: Stage Drop-off")
    plt.xlabel("Funnel Stage")
    plt.ylabel("Users Remaining")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
