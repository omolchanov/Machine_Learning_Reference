{{
    config(
        materialized='table',
    )
}}

with facts as (
    select
        sum(line_total) as total_revenue,
        count(order_id) as order_line_count,
        coalesce(sum(quantity), 0) as units_sold
    from {{ ref('fct_orders') }}
),

scalars as (
    select
        'total_revenue' as metric_name,
        cast(total_revenue as decimal(14, 2)) as metric_value,
        cast(null as date) as metric_date
    from facts

    union all

    select
        'order_line_count',
        cast(order_line_count as decimal(14, 2)),
        null
    from facts

    union all

    select
        'units_sold',
        cast(units_sold as decimal(14, 2)),
        null
    from facts

    union all

    select
        'average_order_value',
        cast(total_revenue / nullif(order_line_count, 0) as decimal(14, 2)),
        null
    from facts
),

daily as (
    select
        'daily_revenue' as metric_name,
        cast(total_revenue as decimal(14, 2)) as metric_value,
        order_date as metric_date
    from {{ ref('daily_sales') }}

    union all

    select
        'daily_active_customers',
        cast(unique_customers as decimal(14, 2)),
        order_date
    from {{ ref('daily_sales') }}
)

select
    metric_name,
    metric_value,
    metric_date,
    current_timestamp as refreshed_at
from scalars

union all

select
    metric_name,
    metric_value,
    metric_date,
    current_timestamp as refreshed_at
from daily