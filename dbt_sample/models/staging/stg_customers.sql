select
    customer_id,
    first_name,
    last_name,
    email,
    city,
    created_at
from {{ source('raw', 'customers') }}