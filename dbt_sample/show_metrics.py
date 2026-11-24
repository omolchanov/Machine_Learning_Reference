"""
Print metric values from the materialized table rpt_metric_values.

Refresh metrics first:
    dbt run --select +rpt_metric_values
"""

import mysql.connector

DB = {
    "host": "127.0.0.1",
    "port": 3307,
    "user": "ml_ref_dev",
    "password": "ml_ref_dev_pass",
    "database": "dbt_sample",
}


def main() -> None:
    conn = mysql.connector.connect(**DB)
    cur = conn.cursor()
    try:
        cur.execute(
            """
            SELECT MAX(refreshed_at) AS last_refresh
            FROM rpt_metric_values
            """
        )
        last = cur.fetchone()[0]
        print(f"Table rpt_metric_values (refreshed_at max = {last})\n")

        cur.execute(
            """
            SELECT metric_name, metric_value
            FROM rpt_metric_values
            WHERE metric_date IS NULL
            ORDER BY metric_name
            """
        )
        print("=== Scalar metrics ===")
        for name, value in cur.fetchall():
            print(f"  {name}: {value}")

        cur.execute(
            """
            SELECT metric_name, metric_date, metric_value
            FROM rpt_metric_values
            WHERE metric_date IS NOT NULL
            ORDER BY metric_name, metric_date
            """
        )
        print("\n=== Daily metrics (sample) ===")
        rows = cur.fetchall()
        for name, dt, value in rows[:5]:
            print(f"  {name} @ {dt}: {value}")
        if len(rows) > 5:
            print(f"  ... {len(rows) - 5} more rows")
    finally:
        cur.close()
        conn.close()


if __name__ == "__main__":
    main()