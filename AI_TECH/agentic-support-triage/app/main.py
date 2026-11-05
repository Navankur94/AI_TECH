"""FastAPI entrypoint. Run: uvicorn app.main:app --host 0.0.0.0 --port 8000"""
from __future__ import annotations

import hmac
import logging
from typing import Literal

from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel, Field, model_validator

from .config import Settings
from .service import InvalidState, TicketNotFound, TriageService

logger = logging.getLogger("triage")


class TicketRequest(BaseModel):
    message: str = Field(min_length=5, max_length=5000)


class ReviewRequest(BaseModel):
    decision: Literal["approved", "edited", "rejected"]
    response: str | None = Field(default=None, max_length=5000)
    reviewer: str = Field(min_length=1, max_length=100)

    @model_validator(mode="after")
    def edited_needs_text(self):
        if self.decision == "edited" and not (self.response and self.response.strip()):
            raise ValueError("response is required when decision is 'edited'")
        return self


def build_service(settings: Settings) -> TriageService:
    from openai import OpenAI

    from .graph import TriageGraph
    from .kb import KnowledgeBase
    from .llm import OpenAIJSON

    client = OpenAI(api_key=settings.openai_api_key or "unset", timeout=settings.request_timeout, max_retries=2)
    graph = TriageGraph(OpenAIJSON(client, settings.chat_model), KnowledgeBase.from_dir(settings.kb_dir),
                        settings.confidence_threshold).build()
    return TriageService(graph)


def create_app(service: TriageService | None = None, settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings.from_env()
    app = FastAPI(title="Agentic Support Triage", version="1.0.0")
    holder: dict[str, TriageService | None] = {"service": service}

    def get_service() -> TriageService:
        if holder["service"] is None:
            holder["service"] = build_service(settings)
        return holder["service"]  # type: ignore[return-value]

    def require_key(x_api_key: str | None = Header(default=None)) -> None:
        if not settings.api_keys:
            raise HTTPException(status_code=503, detail="Server has no API_KEYS configured")
        if not x_api_key or not any(hmac.compare_digest(x_api_key, k) for k in settings.api_keys):
            raise HTTPException(status_code=401, detail="Invalid or missing API key")

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.post("/v1/tickets", dependencies=[Depends(require_key)], status_code=201)
    def create_ticket(payload: TicketRequest):
        try:
            return get_service().submit(payload.message)
        except Exception as exc:
            logger.exception("ticket processing failed")
            raise HTTPException(status_code=502, detail="Ticket processing failed") from exc

    @app.get("/v1/tickets", dependencies=[Depends(require_key)])
    def pending_tickets():
        return {"pending": get_service().pending()}

    @app.get("/v1/tickets/{ticket_id}", dependencies=[Depends(require_key)])
    def get_ticket(ticket_id: str):
        try:
            return get_service().view(ticket_id)
        except TicketNotFound:
            raise HTTPException(status_code=404, detail="Ticket not found")

    @app.post("/v1/tickets/{ticket_id}/review", dependencies=[Depends(require_key)])
    def review_ticket(ticket_id: str, payload: ReviewRequest):
        try:
            return get_service().review(ticket_id, payload.decision, payload.response, payload.reviewer)
        except TicketNotFound:
            raise HTTPException(status_code=404, detail="Ticket not found")
        except InvalidState as exc:
            raise HTTPException(status_code=409, detail=str(exc))

    return app


app = create_app()
