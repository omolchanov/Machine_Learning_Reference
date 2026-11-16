{{
    config(
        unique_key='order_id',
        incremental_strategy='append',
    )
}}

select
    order_id,
    customer_id,
    product_id,
    quantity,
    unit_price,
    line_total,
    order_date,
    status,
    updated_at
from {{ ref('stg_orders') }}
{% if is_incremental() %}
where order_date > (select coalesce(max(order_date), '1970-01-01') from {{ this }})
{% endif %}