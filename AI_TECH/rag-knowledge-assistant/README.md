# RAG Knowledge Assistant

A FastAPI service that answers questions from your own documents and shows where each answer came from.
Upload `.md`, `.txt` or `.pdf` files, then ask questions. Answers carry citations, and when the documents
do not contain the answer the service says so instead of guessing.

## How it works

```text
upload -> chunk by heading (with overlap) -> embed (OpenAI) -> ChromaDB
ask    -> embed question -> vector search (3x top_k) -> lexical rerank -> relevance floor
       -> LLM answers from numbered sources -> citations checked -> response
```

Guards in the answer path:
- No chunk above `MIN_SCORE` means no LLM call and an honest "not found".
- The model must cite sources as `[n]`. Uncited answers are rejected, and citation numbers that do not
  exist are dropped.
- Retrieved text is marked untrusted in the system prompt (prompt-injection defense).
- Re-uploading a file replaces its chunks, so re-indexing never creates duplicates.

## Run with Docker

```bash
cp .env.example .env            # then set OPENAI_API_KEY and API_KEYS
docker compose up --build       # API on http://localhost:8001, docs at /docs
```

```bash
curl -H "x-api-key: $KEY" -F "file=@data/sample_docs/leave_policy.md" localhost:8001/v1/documents
curl -H "x-api-key: $KEY" -H "content-type: application/json" \
     -d '{"question": "How many unused leave days can I carry forward?"}' localhost:8001/v1/ask
```

## Run locally

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
cp .env.example .env
uvicorn app.main:app --reload
pytest                                              # 29 tests, no API key needed
EMBEDDER=hashing MIN_SCORE=0 python scripts/eval_retrieval.py   # offline retrieval eval
OPENAI_API_KEY=... MIN_SCORE=0 python scripts/eval_retrieval.py # same eval with real embeddings
```

## API

| Method | Path | Purpose |
| --- | --- | --- |
| GET | `/health` | Liveness, no key needed |
| POST | `/v1/documents` | Upload a file (multipart). Replaces the same file name |
| GET | `/v1/documents` | List indexed documents with chunk counts |
| DELETE | `/v1/documents/{source}` | Remove a document |
| POST | `/v1/ask` | `{"question": "...", "top_k": 4}` returns answer, `grounded`, `citations` |

All routes except `/health` require the `X-API-Key` header.

## What has been verified, and what has not

| Verified | How |
| --- | --- |
| Chunking, retrieval, re-index, citation rules, auth, upload limits, error mapping | 29 automated tests |
| OpenAI request and response handling | Real OpenAI SDK against a mocked HTTP transport |
| Retrieval quality on the sample documents | Offline eval with the hashing embedder: hit@3 = 1.00 on 12 questions (word overlap only, an easy corpus) |
| Server starts, uploads and lists documents | Run with uvicorn and real ChromaDB |

| Not verified | What to do |
| --- | --- |
| Calls to the live OpenAI API (embeddings and `gpt-4.1-mini`) | Run the Docker steps above with your key and ask a few questions |
| `docker build` | Built by the GitHub Actions workflow in this repository |
| Answer quality, `MIN_SCORE` calibration, load | Run `scripts/eval_retrieval.py` with real embeddings on your own documents |

## Production gaps

Single-process ChromaDB (use a server or pgvector for several replicas), no per-client rate limits
(see the LLM gateway project), no OCR for scanned PDFs, no answer-quality evaluation beyond retrieval,
and API keys are static values from the environment.
