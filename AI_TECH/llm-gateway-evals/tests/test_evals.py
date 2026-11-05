import json
from pathlib import Path

import pytest

from app.evals.cli import main
from app.evals.quote_verify import verify_findings, verify_quotes
from app.evals.runner import EvalCase, load_cases, percentile, run_eval, to_markdown
from app.evals.scorers import run_check
from app.gateway import GatewayResponse

ROOT = Path(__file__).resolve().parents[1]
GOLDEN = str(ROOT / "data/eval/golden.jsonl")


@pytest.mark.parametrize("check,output,expected", [
    ({"type": "contains", "value": "paris"}, "It is Paris.", True),
    ({"type": "not_contains", "value": "guarantee"}, "returns are guaranteed", False),
    ({"type": "equals", "value": "Negative"}, " negative\n", True),
    ({"type": "regex", "value": r"^\d{4}-\d{2}-\d{2}$"}, "2026-03-05", True),
    ({"type": "max_chars", "value": 5}, "toolong", False),
    ({"type": "json_keys", "value": ["a", "b"]}, '{"a": 1, "b": 2}', True),
    ({"type": "json_keys", "value": ["a", "b"]}, '{"a": 1}', False),
    ({"type": "json_keys", "value": ["a"]}, "not json", False),
    ({"type": "json_keys", "value": ["a"]}, "[1]", False),
    ({"type": "mystery"}, "x", False),
])
def test_scorers(check, output, expected):
    assert run_check(check, output)[0] is expected


def test_percentile_uses_nearest_rank():
    values = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    assert percentile(values, 50) == 50 and percentile(values, 95) == 100 and percentile([], 95) == 0.0


def fake_response(text: str, latency: float = 100.0, cost: float = 0.001) -> GatewayResponse:
    return GatewayResponse(text, "gpt-4.1-mini", 10, 5, cost, latency, False, 1, False)


def test_runner_reports_pass_rate_failures_latency_and_cost():
    cases = [
        EvalCase("a", "p1", [{"type": "contains", "value": "yes"}]),
        EvalCase("b", "p2", [{"type": "contains", "value": "yes"}, {"type": "max_chars", "value": 3}]),
    ]
    replies = {"p1": fake_response("yes", 100), "p2": fake_response("yes indeed", 300)}
    report = run_eval(cases, lambda c: replies[c.prompt])
    assert (report.total, report.passed, report.pass_rate) == (2, 1, 0.5)
    assert report.failures[0].id == "b" and "exceeds" in report.failures[0].reasons[0]
    assert report.p95_latency_ms == 300 and report.total_cost_usd == pytest.approx(0.002)
    assert "| b |" in to_markdown(report)


def test_provider_exception_counts_as_failed_case_not_crash():
    def call(case):
        raise RuntimeError("provider down")

    report = run_eval([EvalCase("a", "p", [])], call)
    assert report.pass_rate == 0.0 and report.results[0].error == "provider down"


def test_golden_dataset_is_well_formed():
    cases = load_cases(GOLDEN)
    assert len(cases) >= 6 and all(c.checks and c.mock_response is not None for c in cases)


def test_cli_offline_run_passes_and_writes_json(tmp_path, capsys):
    out = tmp_path / "report.json"
    assert main(["--provider", "fake", "--dataset", GOLDEN, "--json-out", str(out)]) == 0
    assert json.loads(out.read_text())["pass_rate"] == 1.0
    assert "Pass rate" in capsys.readouterr().out


def test_cli_exits_nonzero_below_threshold(tmp_path):
    dataset = tmp_path / "bad.jsonl"
    dataset.write_text(json.dumps({"id": "x", "prompt": "q", "checks": [{"type": "contains", "value": "yes"}],
                                   "mock_response": "no"}) + "\n")
    assert main(["--provider", "fake", "--dataset", str(dataset), "--min-pass-rate", "0.9"]) == 1


TRANSCRIPT = "Advisor: Returns are not guaranteed.\nCustomer: What are   the fees?\nAdvisor: One percent annually."


def test_quote_verification_flags_invented_evidence():
    report = verify_quotes(TRANSCRIPT, ["Returns are not guaranteed.", "what are the fees?", "I guarantee 20% returns"])
    assert len(report.supported) == 2 and report.unsupported == ["I guarantee 20% returns"]
    assert report.support_rate == pytest.approx(2 / 3)


def test_quote_verification_ignores_wrapping_quotes_and_whitespace():
    assert verify_quotes(TRANSCRIPT, ['“One percent   annually.”']).supported


def test_empty_quote_is_unsupported_and_findings_helper_reads_evidence_key():
    findings = [{"evidence": "One percent annually."}, {"evidence": ""}, {"reason": "no evidence field"}]
    report = verify_findings(TRANSCRIPT, findings)
    assert len(report.supported) == 1 and len(report.unsupported) == 2
