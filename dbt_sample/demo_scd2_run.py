"""
Demonstrate SCD Type 2 on customers and products.

MySQL note: dbt snapshot often fails with error 1046 (no default DB inside transactions).
This script applies the same check-strategy logic as snapshots/customers_snapshot.sql
via scd_sync.py, then builds marts with dbt run.

Steps:
  1. Optional --reset-source (Alice back to Austin, mouse price to 29.99)
  2. Sync snapshots from raw (baseline)
  3. Apply optional/demo SQL updates
  4. Sync again (new SCD rows)
  5. dbt run --select customers_scd2+ fct_orders

Pass --try-dbt-snapshot to attempt native dbt snapshot first (works on some warehouses).
"""

import subprocess
import sys
from pathlib import Path

import mysql.connector

from scd_sync import sync_check_snapshot

DB_CONFIG = {
    "host": "127.0.0.1",
    "port": 3307,
    "user": "root",
    "password": "mysql_root_dev",
    "database": "dbt_sample",
}

PROJECT_ROOT = Path(__file__).resolve().parent
OPTIONAL_SQL = (
    PROJECT_ROOT / "migrations" / "optional" / "006_scd2_demo_changes.sql"
)

CUSTOMER_COLS = [
    "customer_id",
    "first_name",
    "last_name",
    "email",
    "city",
    "created_at",
]
CUSTOMER_CHECK = ["first_name", "last_name", "email", "city"]

PRODUCT_COLS = [
    "product_id",
    "product_name",
    "category",
    "unit_price",
    "created_at",
]
PRODUCT_CHECK = ["product_name", "category", "unit_price"]


def run_dbt(args: list[str]) -> None:
    print(f"\n>>> dbt {' '.join(args)}")
    result = subprocess.run(
        ["dbt", *args],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
    )
    print(result.stdout)
    if result.stderr:
        print(result.stderr, file=sys.stderr)
    if result.returncode != 0:
        raise SystemExit(result.returncode)


def reset_source(cursor, connection) -> None:
    cursor.execute("DROP TABLE IF EXISTS customers_snapshot")
    cursor.execute("DROP TABLE IF EXISTS products_snapshot")
    cursor.execute(
        "UPDATE customers SET city = 'Austin' WHERE customer_id = 1"
    )
    cursor.execute(
        "UPDATE products SET unit_price = 29.99 WHERE product_id = 101"
    )
    connection.commit()
    print(
        "Reset snapshots, customer 1 city -> Austin, product 101 unit_price -> 29.99"
    )


def apply_demo_updates(cursor, connection) -> None:
    if not OPTIONAL_SQL.is_file():
        raise SystemExit(f"Missing {OPTIONAL_SQL}")
    script = OPTIONAL_SQL.read_text(encoding="utf-8")
    for statement in filter(None, script.split(";")):
        trimmed = statement.strip()
        if trimmed and not trimmed.upper().startswith("USE "):
            cursor.execute(trimmed)
    connection.commit()
    print(f"Applied dimension changes from {OPTIONAL_SQL.name}")


def sync_all(connection) -> None:
    n1 = sync_check_snapshot(
        connection,
        source_table="dbt_sample.customers",
        snapshot_table="dbt_sample.customers_snapshot",
        unique_key="customer_id",
        check_cols=CUSTOMER_CHECK,
        all_cols=CUSTOMER_COLS,
    )
    n2 = sync_check_snapshot(
        connection,
        source_table="dbt_sample.products",
        snapshot_table="dbt_sample.products_snapshot",
        unique_key="product_id",
        check_cols=PRODUCT_CHECK,
        all_cols=PRODUCT_COLS,
    )
    print(f"scd_sync: customers inserted/initialized={n1}, products={n2}")


def print_alice_orders(cursor) -> None:
    cursor.execute(
        """
        SELECT order_id, order_date, customer_city_at_order, customer_city_current
        FROM fct_orders_scd_context
        WHERE customer_id = 1
        ORDER BY order_id
        LIMIT 10
        """
    )
    rows = cursor.fetchall()
    print("\n--- customer_id=1 (Alice): city at order vs current ---")
    for row in rows:
        print(
            f"  order {row[0]} on {row[1]}: "
            f"at_order={row[2]!r}, current={row[3]!r}"
        )


def print_snapshot_versions(cursor) -> None:
    cursor.execute(
        """
        SELECT customer_id, city, dbt_valid_from, dbt_valid_to
        FROM customers_snapshot
        WHERE customer_id = 1
        ORDER BY dbt_valid_from
        """
    )
    print("\n--- customers_snapshot versions for customer_id=1 ---")
    for row in cursor.fetchall():
        print(f"  city={row[1]!r}, from={row[2]}, to={row[3]}")


def main() -> None:
    if "--try-dbt-snapshot" in sys.argv:
        run_dbt(["snapshot"])

    conn = mysql.connector.connect(**DB_CONFIG)
    cur = conn.cursor()
    try:
        if "--reset-source" in sys.argv:
            reset_source(cur, conn)

        sync_all(conn)
        apply_demo_updates(cur, conn)
        sync_all(conn)

        run_dbt(
            [
                "run",
                "--select",
                "customers_scd2",
                "products_scd2",
                "fct_orders",
                "fct_orders_scd_context",
            ]
        )

        print_snapshot_versions(cur)
        print_alice_orders(cur)
    finally:
        cur.close()
        conn.close()


if __name__ == "__main__":
    main()