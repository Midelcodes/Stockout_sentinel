# llm_number_guardrail

A small, dependency-free Python package that stops an LLM from inventing numbers. Extracted from [Stockout Sentinel](../README.md) so other projects can reuse it, for example a budget brief, a clinic report or a grant summary.

## The pattern

1. Compute the facts in code.
2. Let the LLM phrase them.
3. Check every number in the output against the facts.
4. If the check fails, retry with the list of problems; after the last attempt, use a deterministic fallback text.

## Use

```python
from llm_number_guardrail import allowed_numbers, find_offending, guarded_generate

facts = {"site": "C1399", "stock_end": 1250, "p50": 135, "p90": 190}
allowed = allowed_numbers(facts)

def check(draft):
    return [f"number not in facts: {n}" for n in find_offending(draft, allowed, "en")]

result = guarded_generate(
    generate=my_llm_call,          # takes None or (previous draft, problems)
    check=check,                   # returns a list of problems; empty means pass
    fallback=lambda: my_template(facts),
    max_attempts=3,
)
result["guardrail_status"]         # "passed", "passed_after_retry" or "template_fallback"
```

## What it handles

- English and French number formats: `1,250.5` and `1 250,5`, including no-break and narrow no-break spaces as thousands separators.
- Numbers nested anywhere in the facts, including inside names and dates.
- Repeated offending numbers are reported once, in order, so the retry message stays short.

## What it does not do

It checks that each number exists in the facts, not that it sits beside the right item. Stockout Sentinel adds a project-specific range check for that (`src/guardrail.py`). A passed draft is a checked draft, not a verified one: keep a human in the loop for decisions.

## Tests

`pytest tests/test_number_guardrail.py`
