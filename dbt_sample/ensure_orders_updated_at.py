"""Add orders.updated_at when missing; backfill from order_date."""

import mysql.connector

DB_CONFIG = {
    "host": "127.0.0.1",
    "port": 3307,
    "user": "root",
    "password": "mysql_root_dev",
    "database": "dbt_sample",
}


def main() -> None:
    conn = mysql.connector.connect(**DB_CONFIG)
    cur = conn.cursor()
    try:
        cur.execute("SHOW COLUMNS FROM orders LIKE 'updated_at'")
        if cur.fetchone():
            print("orders.updated_at already present")
            return

        cur.execute("ALTER TABLE orders ADD COLUMN updated_at DATETIME NULL AFTER status")
        cur.execute(
            "UPDATE orders SET updated_at = TIMESTAMP(order_date, '12:00:00') WHERE updated_at IS NULL"
        )
        cur.execute(
            "ALTER TABLE orders MODIFY updated_at DATETIME NOT NULL "
            "DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP"
        )
        try:
            cur.execute("ALTER TABLE orders ADD KEY idx_orders_updated_at (updated_at)")
        except mysql.connector.Error as err:
            if err.errno != 1061:
                raise
        conn.commit()
        print("Added and backfilled orders.updated_at")
    finally:
        cur.close()
        conn.close()


if __name__ == "__main__":
    main()