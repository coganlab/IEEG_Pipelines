{{ name | escape | underline }}

.. automodule:: {{ fullname }}

{% if classes %}
{%- block classes %}
.. autosummary::
    :template: autosummary/class.rst
    :toctree:
{% for class in classes %}
    {{ class }}
{%- endfor %}
{% endblock -%}
{% endif %}
{% if functions %}
{%- block functions %}
.. autosummary::
    :template: autosummary/function.rst
    :toctree:
{% for function in functions %}
    {{ function }}
{%- endfor %}
{% endblock -%}
{% endif %}