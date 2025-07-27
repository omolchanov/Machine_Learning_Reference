"""
LowCardinality(String) column for page, which optimizes storage and speeds up queries on repeated string values.

- LowCardinality(String) stores unique strings in a dictionary internally, storing indexes instead of full strings per
  row.
- Great for columns like page where many repeated values exist.
- Improves memory usage and speeds up grouping and filtering.
"""

from datetime import datetime
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
    # Create table with LowCardinality(String) column for 'page'
    client.command("""
        CREATE TABLE IF NOT EXISTS page_views (
            event_time DateTime,
            user_id UInt64,
            page LowCardinality(String)
        ) ENGINE = MergeTree()
        ORDER BY event_time
        """)

    # Insert sample data
    sample_data = [
        (datetime.strptime('2025-07-27 10:00:00', '%Y-%m-%d %H:%M:%S'), 1, 'home'),
        (datetime.strptime('2025-07-27 10:05:00', '%Y-%m-%d %H:%M:%S'), 2, 'products'),
        (datetime.strptime('2025-07-27 10:10:00', '%Y-%m-%d %H:%M:%S'), 1, 'home'),
        (datetime.strptime('2025-07-27 10:15:00', '%Y-%m-%d %H:%M:%S'), 3, 'contact'),
        (datetime.strptime('2025-07-27 10:20:00', '%Y-%m-%d %H:%M:%S'), 2, 'home'),
    ]

    client.insert(
        table='page_views',
        data=sample_data,
        column_names=['event_time', 'user_id', 'page']
    )

    # Query to verify inserts
    result = client.query("SELECT event_time, user_id, page FROM page_views ORDER BY event_time")

    for row in result.result_rows:
        print(row)


if __name__ == '__main__':
    main()
