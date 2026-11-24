select metric_name
from {{ ref('rpt_metric_values') }}
group by metric_name
having metric_name not in (
    'total_revenue',
    'order_line_count',
    'units_sold',
    'average_order_value',
    'daily_revenue',
    'daily_active_customers'
)

union all

select 'missing_metric' as metric_name
from (select 1) as dummy
where (
    select count(distinct metric_name)
    from {{ ref('rpt_metric_values') }}
    where metric_name in (
        'total_revenue',
        'order_line_count',
        'units_sold',
        'average_order_value',
        'daily_revenue',
        'daily_active_customers'
    )
) < 6