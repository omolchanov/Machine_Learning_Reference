"""
Load offline feature-store views built by dbt (models/features/*).

Usage (from dbt_sample, after `dbt run --select features`):
    python load_features.py
    python load_features.py --table fs_customer_features
"""

from __future__ import annotations

import argparse
from typing import Any

import mysql.connector
import pandas as pd

DB_CONFIG = {
    "host": "127.0.0.1",
    "port": 3307,
    "user": "ml_ref_dev",
    "password": "ml_ref_dev_pass",
    "database": "dbt_sample",
}

FEATURE_TABLES: dict[str, str] = {
    "fs_customer_features": "Customer-level RFM-style (order count, spend)",
    "fs_daily_sales_features": "Daily time-series aggregates",
    "fs_product_features": "Product revenue and units",
    "fs_order_line_features": "Order-line training grain",
    "fs_order_line_pit_features": "Order lines with SCD / point-in-time attrs",
}


def load_table(name: str) -> pd.DataFrame:
    conn = mysql.connector.connect(**DB_CONFIG)
    try:
        return pd.read_sql(f"SELECT * FROM `{name}`", conn)
    finally:
        conn.close()


def load_all() -> dict[str, pd.DataFrame]:
    return {name: load_table(name) for name in FEATURE_TABLES}


def main() -> None:
    parser = argparse.ArgumentParser(description="Load dbt feature-store views")
    parser.add_argument(
        "--table",
        choices=list(FEATURE_TABLES),
        help="Load one feature table (default: all)",
    )
    args = parser.parse_args()

    if args.table:
        df = load_table(args.table)
        print(f"{args.table}: {len(df)} rows, {len(df.columns)} columns")
        print(df.head())
        return

    frames = load_all()
    for name, description in FEATURE_TABLES.items():
        df = frames[name]
        print(f"{name} — {description}")
        print(f"  shape={df.shape}")
    print("\nExample: join customer features to order lines for a simple training frame")
    orders = frames["fs_order_line_features"]
    customers = frames["fs_customer_features"]
    training = orders.merge(
        customers[["customer_id", "feat_order_count", "feat_total_spent"]],
        on="customer_id",
        how="left",
    )
    print(f"  order lines with customer features: {training.shape}")


if __name__ == "__main__":
    main()