-- Optional SCD2 demo: change dimension rows after an initial `dbt snapshot`.
-- Do not run via apply_migrations.py (optional folder). Use demo_scd2_run.py or run manually.

USE dbt_sample;

UPDATE customers
SET city = 'Portland'
WHERE customer_id = 1
  AND city = 'Austin';

UPDATE products
SET unit_price = 34.99
WHERE product_id = 101
  AND unit_price = 29.99;