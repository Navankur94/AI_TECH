"""Ticket lifecycle on top of the graph: redact first, run, pause for review, resume."""
from __future__ import annotations

import uuid

from .redact import redact_pii


class TicketNotFound(Exception):
    pass


class InvalidState(Exception):
    pass


class TriageService:
    def __init__(self, graph) -> None:
        self.graph = graph
        self._ids: list[str] = []  # in-memory index; a database table in production

    @staticmethod
    def _config(ticket_id: str) -> dict:
        return {"configurable": {"thread_id": ticket_id}}

    def submit(self, message: str) -> dict:
        redacted, found = redact_pii(message)  # raw text never enters the graph or its checkpoints
        ticket_id = uuid.uuid4().hex
        self._ids.append(ticket_id)
        self.graph.invoke(
            {"ticket_id": ticket_id, "message": redacted, "pii_found": found, "audit": ["received"]},
            self._config(ticket_id),
        )
        return self.view(ticket_id)

    def view(self, ticket_id: str) -> dict:
        snapshot = self.graph.get_state(self._config(ticket_id))
        values = snapshot.values
        if not values:
            raise TicketNotFound(ticket_id)
        status = "pending_review" if snapshot.next else values.get("status", "processing")
        return {
            "ticket_id": ticket_id,
            "status": status,
            "category": values.get("category"),
            "urgency": values.get("urgency"),
            "risk_flags": values.get("risk_flags", []),
            "pii_found": values.get("pii_found", []),
            "draft": values.get("draft"),
            "final_response": values.get("final_response"),
            "audit": values.get("audit", []),
        }

    def pending(self) -> list[dict]:
        views = [self.view(t) for t in self._ids]
        return [v for v in views if v["status"] == "pending_review"]

    def review(self, ticket_id: str, decision: str, response: str | None, reviewer: str) -> dict:
        config = self._config(ticket_id)
        snapshot = self.graph.get_state(config)
        if not snapshot.values:
            raise TicketNotFound(ticket_id)
        if tuple(snapshot.next) != ("human_review",):
            raise InvalidState("ticket is not waiting for review")
        audit = [*snapshot.values.get("audit", []), f"reviewed:{decision}:{reviewer}"]
        self.graph.update_state(
            config,
            {"review_decision": decision, "review_response": response or "", "reviewer": reviewer, "audit": audit},
        )
        self.graph.invoke(None, config)
        return self.view(ticket_id)
