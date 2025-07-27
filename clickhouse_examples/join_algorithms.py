from datetime import datetime, timedelta
import random

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

    # Create tables
    client.command("""
        CREATE TABLE IF NOT EXISTS user_events (
            event_time DateTime,
            user_id UInt64,
            page LowCardinality(String)
        ) ENGINE = MergeTree()
        ORDER BY event_time;
        """)

    client.command("""
        CREATE TABLE IF NOT EXISTS user_profiles (
            user_id UInt64,
            name String,
            age UInt8
        ) ENGINE = MergeTree()
        ORDER BY user_id;
        """)

    # Create user_logins table for ASOF JOIN
    client.command("""
    CREATE TABLE IF NOT EXISTS user_logins (
        user_id UInt64,
        login_time DateTime,
        login_method String
    ) ENGINE = MergeTree()
    ORDER BY (user_id, login_time)
    """)

    # Insert data
    now = datetime.now()
    pages = ['home', 'about', 'contact', 'product']
    names = ['Alice', 'Bob', 'Carol', 'David']

    user_events = [
        (now - timedelta(minutes=i * 5), random.randint(1, 4), random.choice(pages))
        for i in range(20)
    ]

    user_profiles = [(i, names[i - 1], random.randint(20, 50)) for i in range(1, 5)]

    user_logins = [
        (i, now - timedelta(minutes=random.randint(1, 100)), random.choice(['email', 'google', 'facebook']))
        for i in range(1, 5)
    ]

    client.insert("user_events", user_events, column_names=['event_time', 'user_id', 'page'])
    client.insert("user_profiles", user_profiles, column_names=['user_id', 'name', 'age'])
    client.insert("user_logins", user_logins, column_names=['user_id', 'login_time', 'login_method'])

    # -------------------------------
    # JOIN Queries
    # -------------------------------

    """
    An INNER JOIN returns rows when there is a match in both tables.
    Only those records that satisfy the join condition are included in the result set.
    Efficient, non-deterministic match. If the right-hand table contains multiple matching rows,
    ClickHouse picks any one of them.
    
    ALL INNER JOIN
    Includes all matching rows from the right-hand table, i.e., a true relational join
    
    A LEFT JOIN returns all rows from the left-hand table, and the matched rows from the right-hand table. 
    If there is no match, ClickHouse fills the right-hand columns with NULL.
    
    An ASOF JOIN joins two tables based on the nearest (but not greater) match on a time column.
    """

    queries = {
        "ANY INNER JOIN": """
            SELECT ue.user_id, ue.page, up.name
            FROM user_events AS ue
            ANY INNER JOIN user_profiles AS up ON ue.user_id = up.user_id
            LIMIT 5
        """,
        "ALL INNER JOIN": """
            SELECT ue.user_id, ue.page, up.name
            FROM user_events AS ue
            ALL INNER JOIN user_profiles AS up ON ue.user_id = up.user_id
            LIMIT 5
        """,
        "LEFT JOIN": """
            SELECT ue.user_id, ue.page, up.name
            FROM user_events AS ue
            LEFT JOIN user_profiles AS up ON ue.user_id = up.user_id
            LIMIT 5
        """,
        "ASOF JOIN": """
            SELECT ue.user_id, ue.event_time, ul.login_method
            FROM user_events AS ue
            ASOF JOIN user_logins AS ul
            ON ue.user_id = ul.user_id AND ul.login_time <= ue.event_time
            ORDER BY ue.user_id, ue.event_time
            LIMIT 5
        """
    }

    # Execute and display results
    for join_type, sql in queries.items():
        print(f"\n--- {join_type} ---")
        result = client.query(sql)
        for row in result.result_rows:
            print(row)


if __name__ == '__main__':
    main()
