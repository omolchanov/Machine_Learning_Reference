{% snapshot customers_snapshot %}

{{
    config(
        target_schema='dbt_sample',
        unique_key='customer_id',
        strategy='check',
        check_cols=['first_name', 'last_name', 'email', 'city'],
    )
}}

select
    customer_id,
    first_name,
    last_name,
    email,
    city,
    created_at
from {{ source('raw', 'customers') }}

{% endsnapshot %}