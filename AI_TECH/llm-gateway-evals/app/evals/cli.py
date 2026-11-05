"""Evaluation CLI.

Offline demo:  python -m app.evals.cli --provider fake
Live OpenAI:   OPENAI_API_KEY=... python -m app.evals.cli --provider openai
Exit code 1 when the pass rate is below --min-pass-rate, so CI can gate on it.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from ..config import Settings
from ..gateway import Gateway
from ..providers import ChatRequest, FakeProvider, OpenAIProvider
from ..usage import UsageStore
from .runner import EvalCase, load_cases, run_eval, to_markdown


def build_gateway(provider_name: str, settings: Settings, cases: list[EvalCase]) -> Gateway:
    if provider_name == "fake":
        mocks = {c.prompt: c.mock_response or "" for c in cases}
        provider = FakeProvider(responder=lambda req: mocks.get(req.messages[-1]["content"], ""))
    else:
        from openai import OpenAI

        provider = OpenAIProvider(OpenAI(api_key=settings.openai_api_key, timeout=settings.request_timeout, max_retries=0))
    return Gateway(provider, settings, UsageStore(":memory:"))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run an LLM evaluation set")
    parser.add_argument("--dataset", default="data/eval/golden.jsonl")
    parser.add_argument("--provider", choices=["openai", "fake"], default="openai")
    parser.add_argument("--model", default=None)
    parser.add_argument("--min-pass-rate", type=float, default=0.9)
    parser.add_argument("--json-out", default=None)
    args = parser.parse_args(argv)

    settings = Settings.from_env()
    cases = load_cases(args.dataset)
    gateway = build_gateway(args.provider, settings, cases)
    model = args.model or settings.default_model

    def call(case: EvalCase):
        messages = ([{"role": "system", "content": case.system}] if case.system else []) + [
            {"role": "user", "content": case.prompt}]
        return gateway.chat("eval", ChatRequest(messages, model, 0.0, 500, case.json_mode))

    report = run_eval(cases, call)
    print(to_markdown(report))
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(
            {"pass_rate": report.pass_rate, "total": report.total, "p95_latency_ms": report.p95_latency_ms,
             "total_cost_usd": report.total_cost_usd,
             "failures": [{"id": r.id, "reasons": r.reasons} for r in report.failures]}, indent=2))
    return 0 if report.pass_rate >= args.min_pass_rate else 1


if __name__ == "__main__":
    sys.exit(main())
