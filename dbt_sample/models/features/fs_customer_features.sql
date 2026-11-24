select
    customer_id,
    first_name,
    last_name,
    city,
    order_count as feat_order_count,
    total_spent as feat_total_spent
from {{ ref('customer_orders') }}