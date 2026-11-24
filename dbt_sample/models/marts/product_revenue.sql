select
    p.product_id,
    p.product_name,
    p.category,
    count(o.order_id) as order_line_count,
    coalesce(sum(o.quantity), 0) as units_sold,
    coalesce(sum(o.line_total), 0) as total_revenue
from {{ ref('stg_products') }} as p
left join {{ ref('stg_orders') }} as o
    on o.product_id = p.product_id
group by
    p.product_id,
    p.product_name,
    p.category