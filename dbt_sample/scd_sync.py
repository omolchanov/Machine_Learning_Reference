"""
MySQL-friendly SCD2 check snapshot (same idea as dbt snapshot check strategy).
dbt snapshot can fail on MySQL with error 1046 when CTAS runs inside BEGIN without a default DB.
"""

from __future__ import annotations

import hashlib
from typing import Sequence

from mysql.connector.connection import MySQLConnection


def _row_hash(values: Sequence[object]) -> str:
    parts = ["" if v is None else str(v) for v in values]
    return hashlib.md5("|".join(parts).encode("utf-8")).hexdigest()


def _table_exists(cursor, schema: str, table: str) -> bool:
    cursor.execute(
        """
        SELECT COUNT(*) AS cnt
        FROM information_schema.tables
        WHERE table_schema = %s AND table_name = %s
        """,
        (schema, table),
    )
    return cursor.fetchone()["cnt"] > 0


def _ensure_snapshot_table(
    cursor, snapshot_table: str, source_table: str, unique_key: str
) -> None:
    schema, table = snapshot_table.split(".", 1)
    if _table_exists(cursor, schema, table):
        return
    cursor.execute(f"CREATE TABLE {snapshot_table} LIKE {source_table}")
    cursor.execute(
        f"""
        SELECT index_name
        FROM information_schema.statistics
        WHERE table_schema = %s AND table_name = %s AND index_name != 'PRIMARY'
        GROUP BY index_name
        """,
        (schema, table),
    )
    for row in cursor.fetchall():
        index_name = row.get("index_name") or row.get("INDEX_NAME")
        cursor.execute(f"ALTER TABLE {snapshot_table} DROP INDEX `{index_name}`")
    cursor.execute(f"ALTER TABLE {snapshot_table} DROP PRIMARY KEY")
    cursor.execute(
        f"""
        ALTER TABLE {snapshot_table}
            ADD COLUMN scd_surrogate_id INT NOT NULL AUTO_INCREMENT PRIMARY KEY FIRST,
            ADD COLUMN dbt_scd_id CHAR(32) NOT NULL,
            ADD COLUMN dbt_updated_at DATETIME NOT NULL,
            ADD COLUMN dbt_valid_from DATETIME NOT NULL,
            ADD COLUMN dbt_valid_to DATETIME NULL,
            ADD KEY idx_{table}_{unique_key}_valid ({unique_key}, dbt_valid_to)
        """
    )


def _insert_version(
    cursor,
    snapshot_table: str,
    all_cols: Sequence[str],
    row: dict,
    scd_id: str,
) -> None:
    cols = list(all_cols)
    cursor.execute(
        f"""
        INSERT INTO {snapshot_table} (
            {", ".join(cols)}, dbt_scd_id, dbt_updated_at, dbt_valid_from, dbt_valid_to
        )
        VALUES (
            {", ".join("%s" for _ in cols)},
            %s, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP, NULL
        )
        """,
        [row[c] for c in cols] + [scd_id],
    )


def sync_check_snapshot(
    connection: MySQLConnection,
    *,
    source_table: str,
    snapshot_table: str,
    unique_key: str,
    check_cols: Sequence[str],
    all_cols: Sequence[str],
) -> int:
    """
    Ensure snapshot_table exists; close current rows when check_cols change; insert new versions.
    Returns count of newly inserted snapshot rows.
    """
    check_cols = list(check_cols)
    all_cols = list(all_cols)
    cursor = connection.cursor(dictionary=True)

    _ensure_snapshot_table(cursor, snapshot_table, source_table, unique_key)
    connection.commit()

    cursor.execute(f"SELECT {', '.join(all_cols)} FROM {source_table}")
    source_by_key = {row[unique_key]: row for row in cursor.fetchall()}

    cursor.execute(
        f"SELECT * FROM {snapshot_table} WHERE dbt_valid_to IS NULL"
    )
    current_by_key = {row[unique_key]: row for row in cursor.fetchall()}

    inserted = 0
    if not current_by_key and source_by_key:
        for row in source_by_key.values():
            scd_id = _row_hash([row[c] for c in check_cols])
            _insert_version(cursor, snapshot_table, all_cols, row, scd_id)
            inserted += 1
        connection.commit()
        cursor.close()
        return inserted

    for key, src in source_by_key.items():
        src_hash = _row_hash([src[c] for c in check_cols])
        current = current_by_key.get(key)
        if current is None:
            _insert_version(cursor, snapshot_table, all_cols, src, src_hash)
            inserted += 1
            continue

        cur_hash = _row_hash([current[c] for c in check_cols])
        if cur_hash == src_hash:
            continue

        cursor.execute(
            f"""
            UPDATE {snapshot_table}
            SET dbt_valid_to = CURRENT_TIMESTAMP, dbt_updated_at = CURRENT_TIMESTAMP
            WHERE {unique_key} = %s AND dbt_valid_to IS NULL
            """,
            (key,),
        )
        _insert_version(cursor, snapshot_table, all_cols, src, src_hash)
        inserted += 1

    connection.commit()
    cursor.close()
    return inserted