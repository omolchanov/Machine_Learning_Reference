select
    c.customer_id,
    c.first_name,
    c.last_name,
    c.city,
    count(o.order_id) as order_count,
    coalesce(sum(o.line_total), 0) as total_spent
from {{ ref('stg_customers') }} as c
left join {{ ref('stg_orders') }} as o
    on o.customer_id = c.customer_id
group by
    c.customer_id,
    c.first_name,
    c.last_name,
    c.city