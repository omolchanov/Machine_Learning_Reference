{% snapshot products_snapshot %}

{{
    config(
        target_schema='dbt_sample',
        unique_key='product_id',
        strategy='check',
        check_cols=['product_name', 'category', 'unit_price'],
    )
}}

select
    product_id,
    product_name,
    category,
    unit_price,
    created_at
from {{ source('raw', 'products') }}

{% endsnapshot %}