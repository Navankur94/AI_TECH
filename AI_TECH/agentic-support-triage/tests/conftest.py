from pathlib import Path

import pytest

from app.graph import CLASSIFY_SYSTEM, TriageGraph
from app.kb import KnowledgeBase
from app.service import TriageService

KB_DIR = Path(__file__).resolve().parents[1] / "data" / "kb"


class ScriptedLLM:
    """Returns scripted JSON for the classify and draft steps and records every prompt it sees."""

    def __init__(self, classify=None, draft=None, fail_classify=False):
        self.classify = classify or {"category": "billing", "urgency": "normal"}
        self.draft = draft or {"response": "Statements are emailed on the 5th of each month.", "confidence": 0.9}
        self.fail_classify = fail_classify
        self.prompts: list[str] = []

    def complete_json(self, system: str, user: str) -> dict:
        self.prompts.append(system + "\n" + user)
        if system == CLASSIFY_SYSTEM:
            if self.fail_classify:
                raise RuntimeError("model unavailable")
            return self.classify
        return self.draft


def make_service(llm: ScriptedLLM, threshold: float = 0.6) -> TriageService:
    graph = TriageGraph(llm, KnowledgeBase.from_dir(KB_DIR), threshold).build()
    return TriageService(graph)


@pytest.fixture
def llm():
    return ScriptedLLM()


@pytest.fixture
def service(llm):
    return make_service(llm)
