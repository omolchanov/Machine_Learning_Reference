select
    product_id,
    product_name,
    category,
    unit_price,
    created_at
from {{ source('raw', 'products') }}