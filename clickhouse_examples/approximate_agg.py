"""
Approximate aggregation functions are a core strength of ClickHouse — giving you blazing-fast results
on massive datasets, with just a tiny sacrifice in accuracy (typically <1% error).
"""

from clickhouse_connect import get_client

from web_analytics import profile_performance

client = get_client(
    host='localhost',
    port=8123,
    username='testuser',
    password='testuser',
    database='test'
)

print(client.ping())


@profile_performance
def main():

    # Create a sample table
    client.command("""
    CREATE TABLE IF NOT EXISTS user_events (
        event_time DateTime,
        user_id UInt64,
        page LowCardinality(String)
    ) ENGINE = MergeTree()
    ORDER BY event_time
    """)

    # Insert a large dataset
    from datetime import datetime, timedelta
    import random

    now = datetime(2025, 7, 27, 10)
    data = [
        (now + timedelta(seconds=i), random.randint(1, 1000), random.choice(['home', 'products', 'about', 'contact']))
        for i in range(10000)
    ]

    client.insert(
        table='user_events',
        data=data,
        column_names=['event_time', 'user_id', 'page']
    )

    # Run approximate distinct count query
    result = client.query("SELECT uniqCombined(user_id) FROM user_events")

    # Output result
    print("\nApproximate Unique Users:", result.result_rows[0][0])

    # Calculate session-like quantiles by approximating sessions using user_id + hour
    query = """
    SELECT
        quantile(0.5)(session_length) AS median_session_length,
        quantile(0.9)(session_length) AS p90_session_length,
        quantile(0.99)(session_length) AS p99_session_length
    FROM (
        SELECT
            user_id,
            toStartOfHour(event_time) AS hour,
            max(event_time) - min(event_time) AS session_length
        FROM user_events
        GROUP BY user_id, hour
    )
    """

    # Display result
    result = client.query(query)

    print("\nApproximate Session Length Quantiles (using hour as session key):")
    for col, val in zip(result.column_names, result.result_rows[0]):
        print(f"{col}: {val:.2f}")


if __name__ == '__main__':
    main()
