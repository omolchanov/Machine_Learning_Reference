select
    order_id,
    customer_id,
    product_id,
    quantity,
    unit_price,
    line_total,
    order_date,
    status
from {{ ref('stg_orders') }}
where status = 'shipped'