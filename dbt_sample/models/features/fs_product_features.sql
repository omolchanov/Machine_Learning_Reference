select
    product_id,
    product_name,
    category,
    order_line_count as feat_order_line_count,
    units_sold as feat_units_sold,
    total_revenue as feat_total_revenue
from {{ ref('product_revenue') }}