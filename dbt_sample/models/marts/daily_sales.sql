select
    o.order_date,
    count(o.order_id) as order_line_count,
    count(distinct o.customer_id) as unique_customers,
    coalesce(sum(o.quantity), 0) as units_sold,
    coalesce(sum(o.line_total), 0) as total_revenue
from {{ ref('stg_orders') }} as o
group by
    o.order_date