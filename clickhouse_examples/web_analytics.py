from uuid import uuid4
import random
import datetime
import time
from functools import wraps
import tracemalloc
import pprint

from clickhouse_connect import get_client


client = get_client(
    host='localhost',
    port=8123,
    username='testuser',
    password='testuser',
    database='test'
)

print(client.ping())


def profile_performance(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Start measuring time and memory
        tracemalloc.start()
        start_time = time.perf_counter()

        result = func(*args, **kwargs)

        end_time = time.perf_counter()
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        # Report
        print(f"\n[{func.__name__}] Time: {(end_time - start_time):.4f}s")
        print(f"[{func.__name__}] Memory: Current={current / 1024 / 1024:.3f} MB, Peak={peak / 1024 / 1024:.1f} MB")

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


@profile_performance
def total_events():
    return client.query('SELECT COUNT(*) FROM web_events').result_rows[0][0]


@profile_performance
def unique_users():
    return client.query('SELECT COUNT(DISTINCT user_id) FROM web_events').result_rows[0][0]


@profile_performance
def events_per_page():
    result = client.query('''
        SELECT page, COUNT(*) AS event_count
        FROM web_events
        GROUP BY page
        ORDER BY event_count DESC
    ''')
    return result.result_rows


@profile_performance
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


@profile_performance
def first_event_time():
    return client.query('SELECT MIN(event_time) FROM web_events').result_rows[0][0]


@profile_performance
def last_event_time():
    return client.query('SELECT MAX(event_time) FROM web_events').result_rows[0][0]


@profile_performance
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


@profile_performance
def get_unique_users_per_day():
    query = """
        SELECT
            toStartOfHour(event_time) AS hour,
            uniqExact(user_id) AS exact_unique_users,
            uniqCombined(user_id) AS approx_unique_users
        FROM web_events
        GROUP BY hour
        ORDER BY hour
    """
    result = client.query(query)

    return [
        {
            'hour': row[0].strftime("%H") if hasattr(row[0], "strftime") else str(row[0]),
            'exact_unique_users': row[1],
            'approx_unique_users': row[2],
        }
        for row in result.result_rows
    ]


@profile_performance
def get_running_total_events_by_hour():
    query = """
        SELECT
            hour,
            sum(event_count) OVER (ORDER BY hour ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS 
            running_total_events
        FROM (
            SELECT toStartOfHour(event_time) AS hour, count(*) AS event_count
            FROM web_events
            GROUP BY hour
        )
        ORDER BY hour
    """
    result = client.query(query)

    return [
        {
            'hour': row[0].strftime("%H") if hasattr(row[0], "strftime") else str(row[0]),
            'running_total_events': row[1],
        }
        for row in result.result_rows
    ]


@profile_performance
def get_session_length_distribution():
    query = """
        SELECT
            quantile(0.5)(session_length) AS median_session_length,
            quantile(0.9)(session_length) AS p90_session_length,
            quantile(0.99)(session_length) AS p99_session_length
        FROM (
            SELECT
                session_id,
                max(toDateTime(event_time)) - min(toDateTime(event_time)) AS session_length
            FROM web_events
            GROUP BY session_id
        )
    """
    result = client.query(query)
    row = result.result_rows[0]
    return {
        'median_session_length': row[0],
        'p90_session_length': row[1],
        'p99_session_length': row[2],
    }


@profile_performance
def get_top_pages_last_2_hours():
    query = """
        SELECT
            page,
            count(*) AS events
        FROM web_events
        WHERE toDateTime(event_time) >= now() - INTERVAL 2 HOUR
          AND is_logged_in = 1
        GROUP BY page
        ORDER BY events DESC
        LIMIT 5
    """
    result = client.query(query)
    return [
        {'page': row[0], 'events': row[1]}
        for row in result.result_rows
    ]


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

    print("\nUnique users per hour:")
    pprint.pprint(get_unique_users_per_day())

    print("\nTotal events by hour:")
    pprint.pp(get_running_total_events_by_hour())

    print(get_session_length_distribution())
    print(get_top_pages_last_2_hours())
