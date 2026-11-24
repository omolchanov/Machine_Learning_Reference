select
    order_id,
    customer_id,
    product_id,
    order_date as feat_order_date,
    order_unit_price as feat_order_unit_price,
    customer_city_current as feat_customer_city_current,
    customer_city_at_order as feat_customer_city_at_order,
    catalog_unit_price_current as feat_catalog_unit_price_current,
    catalog_unit_price_at_order as feat_catalog_unit_price_at_order
from {{ ref('fct_orders_scd_context') }}