select
    order_id,
    customer_id,
    product_id,
    quantity,
    unit_price,
    quantity * unit_price as line_total,
    order_date,
    status
from {{ source('raw', 'orders') }}