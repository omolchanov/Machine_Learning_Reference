from uuid import uuid4
import random
import datetime
import time
from functools import wraps

from clickhouse_connect import get_client


client = get_client(
    host='localhost',
    port=8123,
    username='testuser',
    password='testuser',
    database='test'
)

print(client.ping())


def timeit(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        duration = end - start
        print(f"\nFunction '{func.__name__}' executed in {duration:.4f} seconds")
        return result
    return wrapper


def create_table():
    sql = """
        CREATE TABLE IF NOT EXISTS web_events (
        user_id UInt32,
        event_time DateTime,
        page String,
        session_id UUID,
        load_time Float64,           -- e.g., page load time in seconds
        is_logged_in UInt8,          -- 0 or 1 for false/true (Boolean stored as UInt8)
        referrer Nullable(String)    -- optional referrer URL
        ) ENGINE = MergeTree()
        ORDER BY event_time;
    """
    client.command(sql)


def ingest_data():
    pages = ['/home', '/about', '/product', '/contact', '/blog']
    referrers = [None, 'https://google.com', 'https://facebook.com', 'https://twitter.com', None]
    start_time = datetime.datetime(2025, 7, 1)

    batch_size = 1000
    total_rows = 10000

    for offset in range(0, total_rows, batch_size):
        values = []
        for i in range(offset, offset + batch_size):
            user_id = random.randint(1, 1000)
            event_time = (start_time + datetime.timedelta(seconds=i * 5)).strftime('%Y-%m-%d %H:%M:%S')
            page = random.choice(pages)
            session_id = str(uuid4())
            load_time = round(random.uniform(0.1, 5.0), 3)  # float seconds with milliseconds
            is_logged_in = random.randint(0, 1)  # boolean as UInt8
            referrer = random.choice(referrers)
            referrer_val = f"'{referrer}'" if referrer else "NULL"

            values.append(
                f"({user_id}, '{event_time}', '{page}', '{session_id}', {load_time}, {is_logged_in}, {referrer_val})")

        client.command(f"""
            INSERT INTO web_events (user_id, event_time, page, session_id, load_time, is_logged_in, referrer)
            VALUES {','.join(values)}
        """)
        print(f"Inserted rows {offset + 1} to {offset + batch_size}")


@timeit
def total_events():
    return client.query('SELECT COUNT(*) FROM web_events').result_rows[0][0]


@timeit
def unique_users():
    return client.query('SELECT COUNT(DISTINCT user_id) FROM web_events').result_rows[0][0]


@timeit
def events_per_page():
    result = client.query('''
        SELECT page, COUNT(*) AS event_count
        FROM web_events
        GROUP BY page
        ORDER BY event_count DESC
    ''')
    return result.result_rows


@timeit
def average_events_per_user():
    result = client.query('''
        SELECT AVG(events)
        FROM (
            SELECT user_id, COUNT(*) AS events
            FROM web_events
            GROUP BY user_id
        )
    ''')
    return result.result_rows[0][0]


@timeit
def first_event_time():
    return client.query('SELECT MIN(event_time) FROM web_events').result_rows[0][0]


@timeit
def last_event_time():
    return client.query('SELECT MAX(event_time) FROM web_events').result_rows[0][0]


@timeit
def top_active_users(limit=5):
    query = f'''
        SELECT user_id, COUNT(*) AS event_count
        FROM web_events
        GROUP BY user_id
        ORDER BY event_count DESC
        LIMIT {limit}
    '''
    result = client.query(query)
    return result.result_rows


if __name__ == '__main__':
    # Prepare sample data
    # create_table()
    # ingest_data()

    # Perform agg queries
    print("Total events:", total_events())
    print("Unique users:", unique_users())
    print("Events per Page:", events_per_page())
    print("Average Events per User:", average_events_per_user())
    print("First Event Time:", first_event_time())
    print("Last Event Time:", last_event_time())
    print("Top Active Users:", top_active_users())
