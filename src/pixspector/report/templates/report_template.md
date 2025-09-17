# pixspector Forensic Report

**File:** {{ input }}
**SHA-256:** {{ sha256 }}

**Suspicion Index:** {{ suspicion_index }} ({{ bucket_label }})

---

## Evidence

{% if evidence %}
{% for e in evidence -%}

- **{{ e.key }}** ({{ e.weight }}): {{ e.rationale }}{% if e.value %} (value={{ e.value }}){% endif %}
  {% endfor %}
  {% else %}
  No strong forensic cues found.
  {% endif %}

---

## Notes

{% for n in notes -%}

- {{ n }}
  {% endfor %}
