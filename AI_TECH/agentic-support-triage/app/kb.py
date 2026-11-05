"""Tiny keyword knowledge base over Markdown FAQ files (swap for a vector store at scale)."""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

_STOP = frozenset("a an and are as at be by can do does for from have how i in is it me my of on or the to what when where which will with you your".split())
_HEADING = re.compile(r"^#{1,6}\s+(.+?)\s*$", re.MULTILINE)


def _tokens(text: str) -> set[str]:
    return {t for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in _STOP}


@dataclass(frozen=True)
class Entry:
    title: str
    text: str
    source: str


class KnowledgeBase:
    def __init__(self, entries: list[Entry]) -> None:
        self.entries = entries

    @classmethod
    def from_dir(cls, folder: str | Path) -> "KnowledgeBase":
        entries: list[Entry] = []
        for path in sorted(Path(folder).glob("*.md")):
            text = path.read_text(encoding="utf-8")
            matches = list(_HEADING.finditer(text))
            for i, m in enumerate(matches):
                end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
                body = text[m.end():end].strip()
                if body:
                    entries.append(Entry(m.group(1), body, path.name))
        return cls(entries)

    def search(self, query: str, k: int = 2) -> list[Entry]:
        q = _tokens(query)
        scored = []
        for entry in self.entries:
            overlap = len(q & (_tokens(entry.title) | _tokens(entry.text)))
            if overlap:
                scored.append((overlap, entry))
        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [entry for _, entry in scored[:k]]
