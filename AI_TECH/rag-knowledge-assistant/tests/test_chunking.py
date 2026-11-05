import pytest

from app.chunking import chunk_document, split_sections


def test_sections_follow_headings():
    text = "Intro text\n\n# Alpha\nfirst body\n\n## Beta\nsecond body"
    sections = split_sections(text)
    assert [s[0] for s in sections] == ["Introduction", "Alpha", "Beta"]
    assert sections[2][1] == "second body"


def test_chunks_never_mix_sections():
    text = "# One\naaa aaa\n\n# Two\nbbb bbb"
    chunks = chunk_document("doc.md", text, max_chars=200, overlap=20)
    assert [c.section for c in chunks] == ["One", "Two"]
    assert "bbb" not in chunks[0].text


def test_chunk_size_is_respected_and_overlap_carries_text():
    paragraphs = [f"Sentence number {i} has some filler words to take space." for i in range(30)]
    text = "# Long\n" + "\n\n".join(paragraphs)
    chunks = chunk_document("doc.md", text, max_chars=300, overlap=60)
    assert len(chunks) > 1
    assert all(len(c.text) <= 300 for c in chunks)
    last_words = chunks[0].text.split()[-3:]
    assert all(word in chunks[1].text for word in last_words)


def test_oversized_paragraph_is_hard_split():
    text = "# Big\n" + "word " * 400
    chunks = chunk_document("doc.md", text, max_chars=200, overlap=20)
    assert len(chunks) > 5
    assert all(len(c.text) <= 200 for c in chunks)


def test_ids_are_deterministic_and_unique():
    text = "# A\n" + "x " * 300 + "\n\n# B\n" + "y " * 300
    first = chunk_document("doc.md", text, 200, 20)
    second = chunk_document("doc.md", text, 200, 20)
    assert [c.id for c in first] == [c.id for c in second]
    assert len({c.id for c in first}) == len(first)


def test_empty_text_gives_no_chunks():
    assert chunk_document("doc.md", "   \n\n") == []


@pytest.mark.parametrize("max_chars,overlap", [(10, 0), (200, 200), (200, -1)])
def test_invalid_parameters_are_rejected(max_chars, overlap):
    with pytest.raises(ValueError):
        chunk_document("doc.md", "# A\nbody", max_chars, overlap)
