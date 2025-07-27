"""
Stores arrays and nested structures directly.
"""

from clickhouse_connect import get_client

client = get_client(
    host='localhost',
    port=8123,
    username='testuser',
    password='testuser',
    database='test'
)

print(client.ping())


if __name__ == '__main__':

    # Create table with Array(String) column
    client.command("""
    CREATE TABLE IF NOT EXISTS user_activities (
        user_id UInt64,
        pages Array(String)
    ) ENGINE = MergeTree()
    ORDER BY user_id
    """)

    # Insert sample data
    sample_data = [
        (1, ['home', 'about', 'contact']),
        (2, ['home', 'products']),
        (3, ['contact']),
    ]

    client.insert(
        table='user_activities',
        data=sample_data,
        column_names=['user_id', 'pages']
    )

    # Query using arrayJoin to flatten pages arrays
    result = client.query("""
    SELECT user_id, arrayJoin(pages) AS page FROM user_activities
    ORDER BY user_id
    """)

    for row in result.result_rows:
        print(f"user_id: {row[0]}, page: {row[1]}")

    # Query without flattening
    result = client.query("""
    SELECT user_id, pages FROM user_activities ORDER BY user_id
    """)

    for row in result.result_rows:
        print(f"user_id: {row[0]}, pages: {row[1]}")
