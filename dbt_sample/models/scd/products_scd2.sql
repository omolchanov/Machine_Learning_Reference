select
    product_id,
    product_name,
    category,
    unit_price,
    created_at,
    dbt_scd_id,
    dbt_updated_at,
    dbt_valid_from,
    dbt_valid_to
from {{ source('scd', 'products_snapshot') }}