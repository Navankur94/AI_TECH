# Agentic Support Triage (LangGraph)

A FastAPI service that triages customer support messages for a wealth-management platform with a
LangGraph state graph. Low-risk questions are answered automatically; anything risky pauses for a
human reviewer who can approve, edit or reject the draft.

## The graph

```text
classify -> retrieve (FAQ) -> draft -> risk_check --clean--------------> finalize -> END
                                            \--any risk flag--> [PAUSE] human_review -> END
```

- The graph pauses with `interrupt_before=["human_review"]`. State is kept by a checkpointer, so a
  reviewer can resume the same ticket later through the API.
- Risk rules are plain code, so the model cannot talk its way past the gate: high-risk categories
  (fraud, complaint, investment query), promise language such as "guaranteed returns", low model
  confidence, high urgency, and any model error all force review.
- Fail safe: if the model is down or returns garbage, the ticket goes to a human, not to the customer.
- PII (email, PAN, Aadhaar, mobile numbers, Luhn-valid card numbers) is masked before the message
  enters the graph, so raw values never reach the model or the stored state.

## Run with Docker

```bash
cp .env.example .env            # then set OPENAI_API_KEY and API_KEYS
docker compose up --build       # API on http://localhost:8002, docs at /docs
```

```bash
curl -H "x-api-key: $KEY" -H "content-type: application/json" \
     -d '{"message": "When is my monthly statement emailed?"}' localhost:8002/v1/tickets
curl -H "x-api-key: $KEY" localhost:8002/v1/tickets                      # tickets awaiting review
curl -H "x-api-key: $KEY" -H "content-type: application/json" \
     -d '{"decision": "approved", "reviewer": "meera"}' localhost:8002/v1/tickets/<id>/review
```

## Run locally

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
cp .env.example .env
uvicorn app.main:app --reload
pytest                           # 31 tests, no API key needed
```

## API

| Method | Path | Purpose |
| --- | --- | --- |
| GET | `/health` | Liveness, no key needed |
| POST | `/v1/tickets` | Submit a message. Returns status `auto_resolved` or `pending_review` |
| GET | `/v1/tickets` | Tickets waiting for review |
| GET | `/v1/tickets/{id}` | Ticket view with category, risk flags, draft, audit trail |
| POST | `/v1/tickets/{id}/review` | `approved`, `edited` (needs `response`) or `rejected` (escalates) |

## What has been verified, and what has not

| Verified | How |
| --- | --- |
| Routing for each risk rule, approve, edit and reject paths, fail-safe on model errors | 31 automated tests with a scripted model |
| Raw PII never appears in model prompts or stored state | Test that captures every prompt |
| OpenAI JSON-mode request and parsing | Real OpenAI SDK against a mocked HTTP transport |
| Server behaviour with the model unreachable | Run with uvicorn and an invalid key: the ticket went to `pending_review` with `classifier_error` |

| Not verified | What to do |
| --- | --- |
| Classification and draft quality from the live `gpt-4.1-mini` | Send 20-30 real or realistic tickets and check categories and drafts |
| `docker build` | Built by the GitHub Actions workflow in this repository |

## Production gaps

State is in memory (`MemorySaver`) and the ticket index is a Python list, so tickets are lost on
restart and the service cannot run as several replicas. Replace them with a Postgres checkpointer and a
tickets table. The knowledge base is keyword search over one sample FAQ; use a vector store for a
real one. PII detection is regex-based and misses names and free-text addresses.
