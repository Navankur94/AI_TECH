"""Run a labeled prompt set through the gateway and report pass rate, latency and cost."""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from ..gateway import GatewayResponse
from .scorers import run_check


@dataclass(frozen=True)
class EvalCase:
    id: str
    prompt: str
    checks: list[dict]
    system: str | None = None
    json_mode: bool = False
    mock_response: str | None = None  # used only by the offline fake provider


@dataclass
class CaseResult:
    id: str
    passed: bool
    reasons: list[str]
    latency_ms: float
    cost_usd: float
    error: str | None = None


@dataclass
class EvalReport:
    total: int
    passed: int
    pass_rate: float
    p50_latency_ms: float
    p95_latency_ms: float
    total_cost_usd: float
    results: list[CaseResult] = field(default_factory=list)

    @property
    def failures(self) -> list[CaseResult]:
        return [r for r in self.results if not r.passed]


def load_cases(path: str | Path) -> list[EvalCase]:
    cases = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            cases.append(EvalCase(id=row["id"], prompt=row["prompt"], checks=row["checks"], system=row.get("system"),
                                  json_mode=row.get("json_mode", False), mock_response=row.get("mock_response")))
    return cases


def percentile(values: list[float], pct: float) -> float:
    """Nearest-rank percentile."""
    if not values:
        return 0.0
    ordered = sorted(values)
    rank = max(1, math.ceil(pct / 100 * len(ordered)))
    return ordered[rank - 1]


def run_eval(cases: list[EvalCase], call: Callable[[EvalCase], GatewayResponse]) -> EvalReport:
    results: list[CaseResult] = []
    for case in cases:
        try:
            response = call(case)
        except Exception as exc:  # a provider failure is a failed case, not a crashed run
            results.append(CaseResult(case.id, False, [f"call failed: {exc}"], 0.0, 0.0, error=str(exc)))
            continue
        reasons = []
        for check in case.checks:
            ok, reason = run_check(check, response.text)
            if not ok:
                reasons.append(reason)
        results.append(CaseResult(case.id, not reasons, reasons, response.latency_ms, response.cost_usd))
    latencies = [r.latency_ms for r in results if r.error is None]
    passed = sum(r.passed for r in results)
    return EvalReport(
        total=len(results),
        passed=passed,
        pass_rate=passed / len(results) if results else 0.0,
        p50_latency_ms=round(percentile(latencies, 50), 1),
        p95_latency_ms=round(percentile(latencies, 95), 1),
        total_cost_usd=round(sum(r.cost_usd for r in results), 8),
        results=results,
    )


def to_markdown(report: EvalReport) -> str:
    lines = [
        f"**Pass rate:** {report.passed}/{report.total} ({report.pass_rate:.0%})  ",
        f"**Latency:** p50 {report.p50_latency_ms} ms, p95 {report.p95_latency_ms} ms  ",
        f"**Cost:** ${report.total_cost_usd:.6f}",
        "",
    ]
    if report.failures:
        lines += ["| Case | Reasons |", "| --- | --- |"]
        lines += [f"| {r.id} | {'; '.join(r.reasons)} |" for r in report.failures]
    return "\n".join(lines)
