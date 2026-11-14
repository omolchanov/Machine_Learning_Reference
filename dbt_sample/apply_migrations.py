"""
Apply ordered SQL files from dbt_sample/migrations to local MySQL (Docker).
"""

from pathlib import Path

import mysql.connector

MIGRATIONS_DIR = Path(__file__).resolve().parent / "migrations"

DB_CONFIG = {
    "host": "127.0.0.1",
    "port": 3307,
    "user": "root",
    "password": "mysql_root_dev",
}


def run_migration(connection, sql_path: Path) -> None:
    with sql_path.open(encoding="utf-8") as sql_file:
        script = sql_file.read()
    cursor = connection.cursor()
    for statement in filter(None, script.split(";")):
        trimmed = statement.strip()
        if trimmed:
            cursor.execute(trimmed)
    connection.commit()
    cursor.close()
    print(f"Applied: {sql_path.name}")


def main() -> None:
    migration_files = sorted(MIGRATIONS_DIR.glob("*.sql"))
    if not migration_files:
        raise SystemExit(f"No migrations found in {MIGRATIONS_DIR}")

    connection = mysql.connector.connect(**DB_CONFIG)
    try:
        for path in migration_files:
            run_migration(connection, path)
    finally:
        connection.close()

    verify = mysql.connector.connect(**DB_CONFIG, database="dbt_sample")
    try:
        cursor = verify.cursor()
        for table in ("customers", "products", "orders"):
            cursor.execute(f"SELECT COUNT(*) FROM {table}")
            count = cursor.fetchone()[0]
            print(f"{table}: {count} rows")
    finally:
        verify.close()


if __name__ == "__main__":
    main()