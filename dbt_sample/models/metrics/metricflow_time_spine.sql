{{
    config(
        pre_hook="SET SESSION cte_max_recursion_depth = 5000",
    )
}}

with recursive days as (
    select cast('2023-01-01' as date) as date_day
    union all
    select date_add(date_day, interval 1 day)
    from days
    where date_day < cast('2026-12-31' as date)
)

select date_day
from days