select
    order_id,
    customer_id,
    product_id,
    quantity as feat_quantity,
    unit_price as feat_unit_price,
    line_total as feat_line_total,
    order_date as feat_order_date,
    status as feat_order_status,
    updated_at as feat_updated_at
from {{ ref('fct_orders') }}