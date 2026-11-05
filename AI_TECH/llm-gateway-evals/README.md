# LLM Gateway and Evaluation Harness

Two pieces that every LLM application needs once it leaves the demo stage:

1. **A gateway** in front of the model provider: per-client API keys, rate limits, response caching,
   retries with exponential backoff, optional fallback provider, and a usage log with tokens, cost and
   latency for every request.
2. **An evaluation harness** that runs a labeled prompt set through the gateway and reports pass rate,
   latency percentiles and cost, and exits non-zero below a threshold so CI can block a bad change.
   It also includes a quote verifier that catches invented "evidence" in LLM output.

## Request path

```text
client -> API key -> rate limit -> cache (temperature 0 only) -> provider
                                                     |  retryable error: backoff 0.5s, 1s, ...
                                                     |  still failing: fallback provider (if set)
                                                     v
                                    usage log: tokens, cost, latency, cache hit, attempts
```

Non-retryable errors (bad request, bad key) are not retried and do not trigger the fallback. Provider
error details are logged server-side and not returned to clients.

## Run with Docker

```bash
cp .env.example .env            # then set OPENAI_API_KEY and API_KEYS (name:secret pairs)
docker compose up --build       # API on http://localhost:8003, docs at /docs
```

```bash
curl -H "x-api-key: $KEY" -H "content-type: application/json" \
     -d '{"messages": [{"role": "user", "content": "Say hello"}]}' localhost:8003/v1/chat
curl -H "x-api-key: $KEY" localhost:8003/v1/usage      # your client's requests, tokens, cost
curl -H "x-api-key: $KEY" localhost:8003/metrics       # Prometheus-format counters
```

## Run locally

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
cp .env.example .env
uvicorn app.main:app --reload
pytest                                           # 47 tests, no API key needed
```

## Evaluations

Cases live in `data/eval/golden.jsonl`. Each has a prompt and checks (`contains`, `not_contains`,
`equals`, `regex`, `max_chars`, `json_keys`).

```bash
python -m app.evals.cli --provider fake                      # offline demo using each case's mock_response
OPENAI_API_KEY=... python -m app.evals.cli --provider openai --min-pass-rate 0.9 --json-out report.json
```

The offline run only proves the harness works. It says nothing about model quality; run the `openai`
provider for that.

Quote verification for any output that cites evidence:

```python
from app.evals.quote_verify import verify_findings
report = verify_findings(transcript, analysis["compliance_findings"])   # reads each finding's "evidence"
print(report.unsupported)   # quotes that do not appear in the transcript
```

## What has been verified, and what has not

| Verified | How |
| --- | --- |
| Cost arithmetic, caching and expiry, retries and backoff, fallback, rate limits, model allow-list, auth, validation | 47 automated tests |
| Real OpenAI SDK request building, token mapping, and 429/500/400/401/connection error classification | SDK against a mocked HTTP transport, including recovery from a 429 |
| Eval scorers, percentiles, pass-rate gating, quote verification | Automated tests |
| Server starts; with the provider unreachable `/v1/chat` returns 502 and the failure is logged | Run with uvicorn and an invalid key |

| Not verified | What to do |
| --- | --- |
| Calls to the live OpenAI API, real latency, real token counts | Run the Docker steps with your key, then `python -m app.evals.cli --provider openai` |
| Prices | Defaults are `gpt-4.1-mini` at 0.40 / 1.60 USD per 1M tokens. Check the provider's pricing page and set `PRICING_JSON` |
| `docker build` | Built by the GitHub Actions workflow in this repository |

## Production gaps

Cache and rate limiter are in process memory (use Redis for several replicas). Usage log is SQLite
(use Postgres for several replicas). The fallback is wired in code (`Gateway(fallback=...)`) but not
exposed through environment variables. Backoff has no jitter. Only OpenAI is implemented; the
`Provider` interface is where Azure, Bedrock or Vertex adapters would plug in. There is no streaming.
