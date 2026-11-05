"""Mask personal identifiers before any text reaches a model or a checkpoint.

Patterns cover common Indian identifiers (PAN, Aadhaar, mobile numbers) plus emails and
payment card numbers (Luhn-checked). It is a safety net, not a guarantee: names and free-text
addresses are not detected.
"""
from __future__ import annotations

import re

EMAIL = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
PAN = re.compile(r"\b[A-Z]{5}[0-9]{4}[A-Z]\b")
CARD = re.compile(r"\b(?:\d[ -]?){13,19}\b")
AADHAAR = re.compile(r"(?<!\d)(?<!\d[ -])\d{4}[ -]?\d{4}[ -]?\d{4}(?![ -]?\d)")
PHONE = re.compile(r"(?<!\d)(?:\+91[ -]?)?[6-9]\d{9}(?!\d)")


def _luhn_ok(digits: str) -> bool:
    total = 0
    for i, ch in enumerate(reversed(digits)):
        n = int(ch)
        if i % 2 == 1:
            n *= 2
            if n > 9:
                n -= 9
        total += n
    return total % 10 == 0


def redact_pii(text: str) -> tuple[str, list[str]]:
    """Return (masked text, sorted list of PII types found). Raw values are never returned."""
    found: set[str] = set()

    def sub(pattern: re.Pattern[str], label: str, text: str, check=None) -> str:
        def repl(match: re.Match[str]) -> str:
            if check and not check(match.group(0)):
                return match.group(0)
            found.add(label)
            return f"[{label}]"

        return pattern.sub(repl, text)

    text = sub(EMAIL, "EMAIL", text)
    text = sub(PAN, "PAN", text)
    text = sub(CARD, "CARD", text, check=lambda s: _luhn_ok(re.sub(r"\D", "", s)))
    text = sub(AADHAAR, "AADHAAR", text)
    text = sub(PHONE, "PHONE", text)
    return text, sorted(found)
