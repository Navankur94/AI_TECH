"""FastAPI entrypoint. Run: uvicorn app.main:app --host 0.0.0.0 --port 8000"""
from __future__ import annotations

import hmac
import io
import logging
import os
import re
import time
import uuid
from dataclasses import asdict

from fastapi import Depends, FastAPI, File, Header, HTTPException, Request, UploadFile
from pydantic import BaseModel, Field

from .config import Settings
from .service import RAGService

logger = logging.getLogger("rag")
ALLOWED_EXTENSIONS = {".txt", ".md", ".pdf"}


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=2000)
    top_k: int | None = Field(default=None, ge=1, le=10)


def build_service(settings: Settings) -> RAGService:
    """Wire the real OpenAI + ChromaDB service. Imported lazily so tests never need keys."""
    import chromadb

    from .embeddings import HashingEmbedder, OpenAIEmbedder
    from .llm import OpenAIChat
    from .store import VectorStore

    from openai import OpenAI

    client = OpenAI(api_key=settings.openai_api_key or "unset", timeout=settings.request_timeout, max_retries=2)
    embedder = HashingEmbedder() if settings.embedder == "hashing" else OpenAIEmbedder(client, settings.embedding_model)
    store = VectorStore(chromadb.PersistentClient(path=settings.chroma_dir), settings.collection)
    return RAGService(store, embedder, OpenAIChat(client, settings.chat_model), settings)


def safe_source_name(filename: str) -> str:
    name = os.path.basename(filename or "").strip()
    name = re.sub(r"[^A-Za-z0-9._ -]", "_", name)
    if not name or name.startswith("."):
        raise HTTPException(status_code=400, detail="Invalid file name")
    return name


def extract_text(name: str, data: bytes) -> str:
    if name.lower().endswith(".pdf"):
        from pypdf import PdfReader

        try:
            reader = PdfReader(io.BytesIO(data))
            return "\n\n".join(page.extract_text() or "" for page in reader.pages)
        except Exception as exc:  # corrupt or encrypted PDF
            raise HTTPException(status_code=422, detail=f"Could not read PDF: {exc}") from exc
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise HTTPException(status_code=422, detail="File must be UTF-8 text") from exc


def create_app(service: RAGService | None = None, settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings.from_env()
    app = FastAPI(title="RAG Knowledge Assistant", version="1.0.0")
    holder: dict[str, RAGService | None] = {"service": service}

    def get_service() -> RAGService:
        if holder["service"] is None:
            holder["service"] = build_service(settings)
        return holder["service"]  # type: ignore[return-value]

    def require_key(x_api_key: str | None = Header(default=None)) -> None:
        if not settings.api_keys:
            raise HTTPException(status_code=503, detail="Server has no API_KEYS configured")
        if not x_api_key or not any(hmac.compare_digest(x_api_key, k) for k in settings.api_keys):
            raise HTTPException(status_code=401, detail="Invalid or missing API key")

    @app.middleware("http")
    async def log_requests(request: Request, call_next):
        request_id = request.headers.get("x-request-id") or uuid.uuid4().hex[:12]
        started = time.perf_counter()
        response = await call_next(request)
        response.headers["x-request-id"] = request_id
        logger.info("%s %s -> %s in %.0f ms [%s]", request.method, request.url.path,
                    response.status_code, (time.perf_counter() - started) * 1000, request_id)
        return response

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.post("/v1/documents", dependencies=[Depends(require_key)])
    async def upload_document(file: UploadFile = File(...)):
        name = safe_source_name(file.filename or "")
        if os.path.splitext(name)[1].lower() not in ALLOWED_EXTENSIONS:
            raise HTTPException(status_code=415, detail="Only .txt, .md and .pdf files are supported")
        data = await file.read(settings.max_upload_bytes + 1)
        if len(data) > settings.max_upload_bytes:
            raise HTTPException(status_code=413, detail="File too large")
        chunks = get_service().ingest(name, extract_text(name, data))
        return {"source": name, "chunks": chunks}

    @app.get("/v1/documents", dependencies=[Depends(require_key)])
    def list_documents():
        return {"documents": [{"source": s, "chunks": n} for s, n in get_service().store.sources().items()]}

    @app.delete("/v1/documents/{source}", dependencies=[Depends(require_key)])
    def delete_document(source: str):
        removed = get_service().store.delete_source(source)
        if not removed:
            raise HTTPException(status_code=404, detail="Document not found")
        return {"source": source, "chunks_removed": removed}

    @app.post("/v1/ask", dependencies=[Depends(require_key)])
    def ask(payload: AskRequest):
        try:
            return asdict(get_service().ask(payload.question, payload.top_k))
        except HTTPException:
            raise
        except Exception as exc:
            logger.exception("ask failed")
            raise HTTPException(status_code=502, detail="Upstream model error") from exc

    return app


app = create_app()
