# AI_TECH

Three small, tested GenAI services built on Python, FastAPI and Docker. Each runs on its own and has
its own README, Dockerfile, `docker-compose.yml`, pinned `requirements.txt` and `.env.example`.

| Project | What it shows | Port | Tests |
| --- | --- | --- | --- |
| [`rag-knowledge-assistant`](rag-knowledge-assistant) | RAG with ChromaDB: heading-aware chunking, reranking, cited answers, refusal without evidence, retrieval evaluation | 8001 | 29 |
| [`agentic-support-triage`](agentic-support-triage) | LangGraph agent with a human-approval gate, rule-based risk checks, PII masking, fail-safe on model errors | 8002 | 31 |
| [`llm-gateway-evals`](llm-gateway-evals) | LLMOps: per-client keys, rate limits, caching, retries, cost tracking, evaluation harness with CI gating, quote verification | 8003 | 47 |

## Quick start

```bash
cd rag-knowledge-assistant          # or either of the other two
cp .env.example .env                # set OPENAI_API_KEY and API_KEYS; .env is git-ignored
docker compose up --build
```

Run the tests without any API key:

```bash
cd <project> && pip install -r requirements-dev.txt && pytest
```

## Honest status

| | Status |
| --- | --- |
| Unit and API tests (107 in total) | Pass. They use scripted models and a mocked HTTP transport |
| Live OpenAI calls (`gpt-4.1-mini`, embeddings) | **Not yet run.** Run each project once with your key before describing it as working on OpenAI |
| Docker builds | **Not yet run locally.** The `ci.yml` workflow builds all three on GitHub |
| Production use | Not claimed. Each README lists its production gaps |

Never commit `.env`. If a key is ever committed, revoke it at the provider and create a new one.
See `GIT_PUSH.md` for the push commands.
