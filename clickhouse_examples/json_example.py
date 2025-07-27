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
    # Create table
    client.command("""
    CREATE TABLE IF NOT EXISTS json_events (
        id UInt64,
        json_data String
    ) ENGINE = MergeTree()
    ORDER BY id
    """)

    # Insert data
    json_rows = [
        (1, '{"user": {"id": 123, "name": "Alice"}, "actions": ["login", "purchase"]}'),
        (2, '{"user": {"id": 456, "name": "Bob"}, "actions": ["login"]}'),
        (3, '{"user": {"id": 789, "name": "Charlie"}, "actions": ["login", "browse", "logout"]}')
    ]

    # client.insert("json_events", json_rows, column_names=["id", "json_data"])

    # Query: Extract usernames from JSON
    result = client.query("""
    SELECT 
        id,
        JSONExtract(JSONExtract(json_data, 'user', 'String'), 'name', 'String') AS user_name,
        JSONExtract(JSONExtract(json_data, 'user', 'String'), 'id', 'UInt64') AS user_id
    FROM json_events
    """)

    print("User info extracted from JSON:")
    for row in result.result_rows:
        print(row)

    # Query: Flatten actions array
    flattened = client.query("""
    SELECT 
        id,
        arrayJoin(JSONExtractArrayRaw(json_data, 'actions')) AS action
    FROM json_events
    """)

    print("\nFlattened actions:")
    for row in flattened.result_rows:
        print(row)


if __name__ == '__main__':
    main()
