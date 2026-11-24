{% macro round_money(expr) %}
    round(greatest({{ expr }}, 0), 2)
{% endmacro %}