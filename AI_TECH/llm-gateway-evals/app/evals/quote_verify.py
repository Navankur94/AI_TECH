"""Check that quoted evidence really appears in the source text.

Use it on any LLM output that claims 'evidence' (for example the compliance findings in an
advisor call analysis): a finding whose quote is not in the transcript is a hallucination signal.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

_QUOTES = "\"'“”‘’"


def normalize(text: str) -> str:
    text = text.casefold().strip(_QUOTES + " ")
    return re.sub(r"\s+", " ", text)


@dataclass
class QuoteReport:
    supported: list[str] = field(default_factory=list)
    unsupported: list[str] = field(default_factory=list)

    @property
    def support_rate(self) -> float:
        total = len(self.supported) + len(self.unsupported)
        return len(self.supported) / total if total else 1.0


def verify_quotes(source: str, quotes: list[str]) -> QuoteReport:
    haystack = re.sub(r"\s+", " ", source.casefold())
    report = QuoteReport()
    for quote in quotes:
        needle = normalize(quote)
        (report.supported if needle and needle in haystack else report.unsupported).append(quote)
    return report


def verify_findings(source: str, findings: list[dict], key: str = "evidence") -> QuoteReport:
    """Verify the `key` field of each finding dict (for example {"evidence": "exact quote", ...})."""
    return verify_quotes(source, [str(f.get(key, "")) for f in findings])
