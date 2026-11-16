"""
Insert orders newer than fct_orders max(order_date), then print before/after counts.
Run dbt next:  dbt run --select fct_orders
"""

import subprocess
import sys
from datetime import date, timedelta
from pathlib import Path

import mysql.connector

DB_CONFIG = {
    "host": "127.0.0.1",
    "port": 3307,
    "user": "root",
    "password": "mysql_root_dev",
    "database": "dbt_sample",
}


def counts(cursor) -> tuple[tuple[int, object], tuple[int, object]]:
    cursor.execute("SELECT COUNT(*), MAX(order_date) FROM orders")
    raw = cursor.fetchone()
    cursor.execute("SELECT COUNT(*), MAX(order_date) FROM fct_orders")
    fct = cursor.fetchone()
    return raw, fct


def print_counts(label: str, raw, fct) -> None:
    print(f"--- {label} ---")
    print(f"  raw orders:  {raw[0]} rows, max order_date = {raw[1]}")
    print(f"  fct_orders:  {fct[0]} rows, max order_date = {fct[1]}")


def main() -> None:
    conn = mysql.connector.connect(**DB_CONFIG)
    cur = conn.cursor()
    try:
        raw, fct = counts(cur)
        print_counts("BEFORE", raw, fct)
        gap = raw[0] - fct[0]
        print(f"  gap:         {gap} row(s) in raw not yet in fact (or dates above fact watermark)")

        cur.execute("SELECT COALESCE(MAX(order_id), 1000) FROM orders")
        next_id = cur.fetchone()[0] + 1
        fct_max = fct[1]
        inserts = [
            (next_id, 4, 101, 3, 29.99, str(fct_max), "completed"),
            (next_id + 1, 5, 105, 2, 9.99, str(fct_max), "shipped"),
        ]
        if isinstance(fct_max, date):
            d1 = fct_max + timedelta(days=1)
            d2 = fct_max + timedelta(days=2)
            inserts[0] = (*inserts[0][:5], d1.isoformat(), inserts[0][6])
            inserts[1] = (*inserts[1][:5], d2.isoformat(), inserts[1][6])

        added = 0
        for row in inserts:
            cur.execute(
                """
                INSERT INTO orders (
                    order_id, customer_id, product_id, quantity, unit_price, order_date, status
                )
                SELECT %s, %s, %s, %s, %s, %s, %s FROM DUAL
                WHERE NOT EXISTS (SELECT 1 FROM orders WHERE order_id = %s)
                """,
                (*row, row[0]),
            )
            added += cur.rowcount
        conn.commit()
        print(f"\nInserted {added} new order line(s) with order_date > fact watermark.")

        raw, fct = counts(cur)
        print_counts("AFTER INSERT (fact still stale until dbt incremental run)", raw, fct)
        print(f"  gap:         {raw[0] - fct[0]} row(s) waiting for incremental append")

        if "--run-dbt" in sys.argv:
            root = Path(__file__).resolve().parent
            print("\n>>> dbt run --select fct_orders")
            result = subprocess.run(
                ["dbt", "run", "--select", "fct_orders"],
                cwd=root,
                text=True,
                capture_output=True,
            )
            print(result.stdout)
            if result.stderr:
                print(result.stderr, file=sys.stderr)
            if result.returncode != 0:
                raise SystemExit(result.returncode)

            cur.close()
            conn.close()
            conn = mysql.connector.connect(**DB_CONFIG)
            cur = conn.cursor()
            raw, fct = counts(cur)
            print_counts("AFTER dbt incremental run", raw, fct)
            print(f"  gap:         {raw[0] - fct[0]} (should be 0)")
    finally:
        cur.close()
        conn.close()


if __name__ == "__main__":
    main()