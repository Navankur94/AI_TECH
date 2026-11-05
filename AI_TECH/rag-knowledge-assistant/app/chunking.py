"""Heading-aware chunking with overlap.

Documents are split on Markdown headings first, so a chunk never mixes two
sections. Each section is then packed paragraph by paragraph into windows of at
most ``max_chars``, with the tail of the previous window repeated as overlap.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

_HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*$", re.MULTILINE)


@dataclass(frozen=True)
class Chunk:
    id: str
    source: str
    section: str
    index: int
    text: str


def split_sections(text: str) -> list[tuple[str, str]]:
    """Return (section title, body) pairs. Text before the first heading is 'Introduction'."""
    matches = list(_HEADING.finditer(text))
    if not matches:
        return [("Introduction", text.strip())] if text.strip() else []
    sections: list[tuple[str, str]] = []
    preface = text[: matches[0].start()].strip()
    if preface:
        sections.append(("Introduction", preface))
    for i, match in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[match.end():end].strip()
        if body:
            sections.append((match.group(2), body))
    return sections


def _tail(text: str, overlap: int) -> str:
    """Last ``overlap`` characters, trimmed forward to a word boundary."""
    if overlap <= 0 or len(text) <= overlap:
        return text if overlap > 0 else ""
    tail = text[-overlap:]
    space = tail.find(" ")
    return tail[space + 1:] if space != -1 else tail


def _hard_split(paragraph: str, max_chars: int, overlap: int) -> list[str]:
    pieces, start = [], 0
    step = max(1, max_chars - overlap)
    while start < len(paragraph):
        pieces.append(paragraph[start:start + max_chars].strip())
        start += step
    return [p for p in pieces if p]


def _pack(body: str, max_chars: int, overlap: int) -> list[str]:
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", body) if p.strip()]
    windows: list[str] = []
    current = ""
    for paragraph in paragraphs:
        parts = _hard_split(paragraph, max_chars, overlap) if len(paragraph) > max_chars else [paragraph]
        for part in parts:
            candidate = f"{current}\n\n{part}".strip() if current else part
            if len(candidate) <= max_chars:
                current = candidate
                continue
            if current:
                windows.append(current)
                carried = _tail(current, overlap)
                candidate = f"{carried}\n\n{part}".strip() if carried else part
                current = candidate if len(candidate) <= max_chars else part
            else:
                current = part
    if current:
        windows.append(current)
    return windows


def chunk_document(source: str, text: str, max_chars: int = 800, overlap: int = 120) -> list[Chunk]:
    if max_chars < 50:
        raise ValueError("max_chars must be at least 50")
    if not 0 <= overlap < max_chars:
        raise ValueError("overlap must be >= 0 and smaller than max_chars")
    chunks: list[Chunk] = []
    for title, body in split_sections(text):
        for window in _pack(body, max_chars, overlap):
            index = len(chunks)
            chunks.append(Chunk(id=f"{source}:{index}", source=source, section=title, index=index, text=window))
    return chunks
