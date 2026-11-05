"""Deterministic output checks. Each returns (passed, reason-if-failed)."""
from __future__ import annotations

import json
import re


def run_check(check: dict, output: str) -> tuple[bool, str]:
    kind, value = check.get("type"), check.get("value")
    if kind == "contains":
        return (str(value).lower() in output.lower(), f"missing text: {value!r}")
    if kind == "not_contains":
        return (str(value).lower() not in output.lower(), f"forbidden text present: {value!r}")
    if kind == "equals":
        return (output.strip().lower() == str(value).strip().lower(), f"expected exactly {value!r}")
    if kind == "regex":
        return (re.search(str(value), output) is not None, f"no match for /{value}/")
    if kind == "max_chars":
        return (len(output) <= int(value), f"{len(output)} chars exceeds {value}")
    if kind == "json_keys":
        try:
            data = json.loads(output)
        except json.JSONDecodeError:
            return False, "output is not valid JSON"
        if not isinstance(data, dict):
            return False, "JSON output is not an object"
        missing = [k for k in value if k not in data]
        return (not missing, f"missing JSON keys: {missing}")
    return False, f"unknown check type: {kind!r}"
