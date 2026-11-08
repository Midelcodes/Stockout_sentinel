"""A reusable guardrail for LLM text that must not invent numbers.

The pattern: compute the facts in code, let the LLM only phrase them, then
check every number in its output against the facts. Retry with the list of
problems; if it still fails, fall back to a deterministic text.

This package has no dependencies beyond the standard library and knows nothing
about Stockout Sentinel. Any project can import it:

    from llm_number_guardrail import allowed_numbers, find_offending, guarded_generate
"""
import json
import re
from decimal import Decimal

__all__ = [
    "SPACES", "parse_number", "extract_numbers", "allowed_numbers",
    "find_offending", "guarded_generate",
]

# Thousands separators seen in English and French text: space, no-break space,
# narrow no-break space and thin space.
SPACES = "    "
NUMBER_PATTERNS = {
    "en": re.compile(rf"(?<!\d)\d{{1,3}}(?:[,{SPACES}]\d{{3}})+(?:\.\d+)?(?!\d)|\d+(?:\.\d+)?"),
    "fr": re.compile(rf"(?<!\d)\d{{1,3}}(?:[.{SPACES}]\d{{3}})+(?:,\d+)?(?!\d)|\d+(?:[.,]\d+)?"),
}
DOT_THOUSANDS = re.compile(r"\d{1,3}(?:\.\d{3})+")


def parse_number(token, lang):
    """Read one number token. English decimals use '.', French decimals use ','."""
    token = re.sub(f"[{SPACES}]", "", token)
    if lang == "en":
        return Decimal(token.replace(",", ""))
    if "," in token:
        return Decimal(token.replace(".", "").replace(",", "."))
    if DOT_THOUSANDS.fullmatch(token):
        return Decimal(token.replace(".", ""))
    return Decimal(token)


def extract_numbers(text, lang):
    """Every number in the text as (token as written, value)."""
    return [(m.group(), parse_number(m.group(), lang)) for m in NUMBER_PATTERNS[lang].finditer(text)]


def allowed_numbers(facts):
    """Every number anywhere in the serialized facts, including inside names and dates."""
    serialized = json.dumps(facts, ensure_ascii=False, default=str)
    return {Decimal(token) for token in re.findall(r"\d+(?:\.\d+)?", serialized)}


def find_offending(text, allowed, lang):
    """Numbers in the text that are not in the allowed set, in order, without repeats."""
    offending = []
    for token, value in extract_numbers(text, lang):
        if value not in allowed and token not in offending:
            offending.append(token)
    return offending


def guarded_generate(generate, check, fallback, max_attempts):
    """Generate until a draft passes `check`, feeding its problems back each time.

    generate(feedback) -> draft or None. feedback is None on the first call,
        then (previous draft, list of problems).
    check(draft) -> list of problems; empty means the draft passes.
    fallback() -> deterministic text used when every attempt fails.

    Returns {"text", "guardrail_status", "attempts"} where status is
    "passed", "passed_after_retry" or "template_fallback".
    """
    feedback = None
    for attempt in range(1, max_attempts + 1):
        draft = generate(feedback)
        if not draft:
            continue
        problems = check(draft)
        if not problems:
            status = "passed" if attempt == 1 else "passed_after_retry"
            return {"text": draft, "guardrail_status": status, "attempts": attempt}
        feedback = (draft, problems)
    return {"text": fallback(), "guardrail_status": "template_fallback", "attempts": max_attempts}
