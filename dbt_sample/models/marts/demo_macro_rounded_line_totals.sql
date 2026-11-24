select
    customer_id as id,
    product_id,
    {{ round_money_pkg.round_money('quantity * unit_price') }} as total
from {{ ref('stg_orders') }}