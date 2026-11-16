select
    order_id,
    quantity,
    unit_price,
    line_total
from {{ ref('stg_orders') }}
where line_total < 0
   or quantity <= 0
