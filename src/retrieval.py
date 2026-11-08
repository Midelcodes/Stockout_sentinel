"""Retrieval over the project's own documentation.

Splits markdown files into short passages (up to 3 sentences, each tagged with
its section heading), indexes them with TF-IDF over word unigrams and bigrams,
and returns the best matches with their source file and heading, so every
answer can be traced to a passage. A passage must clear a similarity floor and
contain at least half of the question's key words; otherwise nothing is
returned rather than a weak match.

No LLM is involved: this is the retrieval half of retrieval-augmented
generation, kept separate so its results can be checked on their own.
"""
import re
from dataclasses import dataclass
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCES = ["README.md", "docs/*.md", "llm_number_guardrail/README.md"]
HEADING = re.compile(r"^(#{1,3})\s+(.*)$")
SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z\[`*])")
MAX_SENTENCES = 3


@dataclass
class Passage:
    source: str
    heading: str
    text: str


def _chunks(block):
    """A paragraph as passages of up to MAX_SENTENCES; tables, lists and code stay whole."""
    if block.lstrip().startswith(("|", "```")):
        return [block]
    if block.lstrip().startswith(("- ", "1.")):
        items = re.split(r"\n(?=\s*(?:- |\d+\. ))", block)
        return [" ".join(item.split()) for item in items if item.strip()]
    sentences = SENTENCE_END.split(" ".join(block.split()))
    return [" ".join(sentences[i:i + MAX_SENTENCES]) for i in range(0, len(sentences), MAX_SENTENCES)]


def split_sections(markdown, source):
    """Short passages, each carrying the nearest heading as its title."""
    passages, heading, lines = [], source, []

    def flush():
        for block in re.split(r"\n\s*\n", "\n".join(lines).strip()):
            for chunk in _chunks(block) if block.strip() else []:
                passages.append(Passage(source, heading, chunk.strip()))

    for line in markdown.splitlines():
        match = HEADING.match(line)
        if match:
            flush()
            heading, lines = match.group(2).strip(), []
        else:
            lines.append(line)
    flush()
    return passages


def load_passages(root=ROOT, patterns=DEFAULT_SOURCES):
    passages = []
    for pattern in patterns:
        for path in sorted(root.glob(pattern)):
            text = path.read_text(encoding="utf-8")
            passages += split_sections(text, str(path.relative_to(root)))
    return passages


class Index:
    def __init__(self, passages):
        if not passages:
            raise ValueError("no passages to index")
        self.passages = passages
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, stop_words="english")
        documents = [f"{p.heading}\n{p.text}" for p in passages]
        self.matrix = self.vectorizer.fit_transform(documents)
        self.analyze = self.vectorizer.build_analyzer()
        self.words = [{t for t in self.analyze(d) if " " not in t} for d in documents]

    def key_words(self, query):
        return {t for t in self.analyze(query) if " " not in t}

    def search(self, query, k=3, min_score=0.08, min_coverage=0.5):
        """Top-k passages as (score, Passage), best first; empty if nothing is relevant."""
        words = self.key_words(query)
        if not words:
            return []
        scores = linear_kernel(self.vectorizer.transform([query]), self.matrix).ravel()
        results = []
        for i in scores.argsort()[::-1]:
            if scores[i] < min_score or len(results) == k:
                break
            if len(words & self.words[i]) / len(words) >= min_coverage:
                results.append((float(scores[i]), self.passages[i]))
        return results
