"""SYNTHETIC tests for documentation retrieval."""
from src.retrieval import Index, Passage, load_passages, split_sections

DOC = """# Title
Intro line.

## Limitations
Expiry dates are synthetic. Censored demand understates need.

## Budget
The optimiser maximises stockout risks resolved within a budget ceiling.
"""


def test_sections_split_at_headings():
    passages = split_sections(DOC, "doc.md")
    assert [p.heading for p in passages] == ["Title", "Limitations", "Budget"]
    assert all(p.source == "doc.md" for p in passages)


def test_search_returns_the_matching_section_first():
    index = Index(split_sections(DOC, "doc.md"))
    score, passage = index.search("which risks does the budget optimiser resolve?")[0]
    assert passage.heading == "Budget"
    assert score > 0


def test_unrelated_query_returns_nothing():
    index = Index(split_sections(DOC, "doc.md"))
    assert index.search("zebra migration patterns") == []
    assert index.search("   ") == []


def test_real_docs_index_and_answer_a_limitations_question():
    passages = load_passages()
    assert passages and all(isinstance(p, Passage) for p in passages)
    results = Index(passages).search("are the expiry dates real?")
    assert results
    assert any("SYNTHETIC" in p.text or "synthetic" in p.text.lower() for _, p in results)
