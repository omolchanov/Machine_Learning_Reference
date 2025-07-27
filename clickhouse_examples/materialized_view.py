"""
Mat.views automatically maintain pre-aggregated or transformed data for faster queries.

- The materialized view stores countState() aggregates (a partial aggregation state).
- When querying, countMerge() merges these states to get the final counts.
- The MV updates automatically as new rows are inserted into events.
"""

from clickhouse_connect import get_client
from datetime import datetime, timedelta

client = get_client(
    host='localhost',
    port=8123,
    username='testuser',
    password='testuser',
    database='test'
)

print(client.ping())


if __name__ == '__main__':

    # Create base table `events`
    client.command("""
    CREATE TABLE IF NOT EXISTS events (
        event_time DateTime,
        user_id UInt64,
        page String
    ) ENGINE = MergeTree()
    ORDER BY event_time
    """)

    # Create materialized view to aggregate events by hour
    client.command("""
    CREATE MATERIALIZED VIEW IF NOT EXISTS events_by_hour
    ENGINE = AggregatingMergeTree()
    ORDER BY hour AS
    SELECT
        toStartOfHour(event_time) AS hour,
        countState() AS event_count_state
    FROM events
    GROUP BY hour
    """)

    # Insert sample data into events table
    now = datetime.now()
    sample_data = [
        (now - timedelta(minutes=90), 1, 'home'),
        (now - timedelta(minutes=75), 2, 'about'),
        (now - timedelta(minutes=50), 1, 'home'),
        (now - timedelta(minutes=15), 3, 'contact'),
        (now, 2, 'home'),
    ]

    client.insert(
        table='events',
        data=sample_data,
        column_names=['event_time', 'user_id', 'page']
    )

    # Query the materialized view to get aggregated counts by hour
    result = client.query("""
    SELECT
        hour,
        countMerge(event_count_state) AS event_count
    FROM events_by_hour
    GROUP BY hour
    ORDER BY hour
    """)

    for row in result.result_rows:
        print(f"Hour: {row[0]}, Event Count: {row[1]}")
