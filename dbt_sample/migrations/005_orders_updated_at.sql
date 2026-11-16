-- One-time upgrade for databases created before updated_at was added in 001.
-- Fresh installs (001 → 002) already have the column; skip this file or ignore duplicate-column errors.

USE dbt_sample;

ALTER TABLE orders
  ADD COLUMN updated_at DATETIME NULL AFTER status;

UPDATE orders
SET updated_at = TIMESTAMP(order_date, '12:00:00')
WHERE updated_at IS NULL;

ALTER TABLE orders
  MODIFY updated_at DATETIME NOT NULL
    DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP;

ALTER TABLE orders ADD KEY idx_orders_updated_at (updated_at);