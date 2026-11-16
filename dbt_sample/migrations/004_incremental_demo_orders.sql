-- Optional rows for demonstrating incremental fct_orders (run after 002_seed).
-- Inserts three new order lines on dates after the seed max (2024-04-22).

USE dbt_sample;

INSERT INTO orders (order_id, customer_id, product_id, quantity, unit_price, order_date, status)
SELECT
    order_id,
    customer_id,
    product_id,
    quantity,
    unit_price,
    order_date,
    status
FROM (
    SELECT 1012 AS order_id, 1 AS customer_id, 106 AS product_id, 1 AS quantity, 45.00 AS unit_price, '2024-05-01' AS order_date, 'completed' AS status
    UNION ALL
    SELECT 1013, 2, 103, 1, 24.00, '2024-05-03', 'processing'
    UNION ALL
    SELECT 1014, 3, 104, 2, 12.50, '2024-05-05', 'completed'
) AS new_orders
WHERE NOT EXISTS (SELECT 1 FROM orders o WHERE o.order_id = 1012);