"""Tests for the standalone llm_number_guardrail package. No project imports."""
from decimal import Decimal

from llm_number_guardrail import (
    allowed_numbers,
    extract_numbers,
    find_offending,
    guarded_generate,
    parse_number,
)


def test_english_and_french_number_formats():
    assert parse_number("1,250.5", "en") == Decimal("1250.5")
    assert parse_number("1 250,5", "fr") == Decimal("1250.5")
    assert parse_number("1 250", "fr") == Decimal("1250")
    assert parse_number("1.250", "fr") == Decimal("1250")


def test_extracts_every_number():
    assert [v for _, v in extract_numbers("Stock 12, demand 3.5 to 8.", "en")] == [
        Decimal("12"), Decimal("3.5"), Decimal("8"),
    ]


def test_allowed_numbers_cover_nested_facts():
    allowed = allowed_numbers({"a": [1, {"b": 2.5}], "name": "Site C1399"})
    assert {Decimal("1"), Decimal("2.5"), Decimal("1399")} <= allowed


def test_invented_numbers_are_flagged_once_in_order():
    allowed = {Decimal("10"), Decimal("20")}
    assert find_offending("10 then 99 then 20 then 99 then 7", allowed, "en") == ["99", "7"]


def test_guarded_generate_passes_retries_and_falls_back():
    def check(draft):
        return [] if "ok" in draft else ["not ok"]

    first = guarded_generate(lambda fb: "ok", check, lambda: "fallback", 3)
    assert first == {"text": "ok", "guardrail_status": "passed", "attempts": 1}

    drafts = iter(["bad", "ok"])
    seen = []

    def generate(feedback):
        seen.append(feedback)
        return next(drafts)

    second = guarded_generate(generate, check, lambda: "fallback", 3)
    assert second["guardrail_status"] == "passed_after_retry"
    assert seen[1] == ("bad", ["not ok"])

    third = guarded_generate(lambda fb: "bad", check, lambda: "fallback", 2)
    assert third == {"text": "fallback", "guardrail_status": "template_fallback", "attempts": 2}


def test_empty_drafts_count_as_attempts():
    result = guarded_generate(lambda fb: None, lambda d: [], lambda: "fallback", 2)
    assert result["guardrail_status"] == "template_fallback"
