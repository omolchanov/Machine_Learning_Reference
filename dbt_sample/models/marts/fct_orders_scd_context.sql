select
    o.order_id,
    o.customer_id,
    o.product_id,
    o.order_date,
    o.unit_price as order_unit_price,
    c_cur.city as customer_city_current,
    (
        select sc.city
        from {{ ref('customers_scd2') }} as sc
        where sc.customer_id = o.customer_id
        order by
            case when sc.dbt_valid_to is null then 1 else 0 end,
            sc.dbt_valid_from
        limit 1
    ) as customer_city_at_order,
    p_cur.unit_price as catalog_unit_price_current,
    (
        select sp.unit_price
        from {{ ref('products_scd2') }} as sp
        where sp.product_id = o.product_id
        order by
            case when sp.dbt_valid_to is null then 1 else 0 end,
            sp.dbt_valid_from
        limit 1
    ) as catalog_unit_price_at_order
from {{ ref('fct_orders') }} as o
left join {{ ref('stg_customers') }} as c_cur
    on c_cur.customer_id = o.customer_id
left join {{ ref('stg_products') }} as p_cur
    on p_cur.product_id = o.product_id