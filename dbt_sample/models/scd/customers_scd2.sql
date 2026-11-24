select
    customer_id,
    first_name,
    last_name,
    email,
    city,
    created_at,
    dbt_scd_id,
    dbt_updated_at,
    dbt_valid_from,
    dbt_valid_to
from {{ source('scd', 'customers_snapshot') }}