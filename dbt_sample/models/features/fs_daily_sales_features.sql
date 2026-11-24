select
    order_date as feat_date,
    order_line_count as feat_order_line_count,
    unique_customers as feat_unique_customers,
    units_sold as feat_units_sold,
    total_revenue as feat_total_revenue
from {{ ref('daily_sales') }}