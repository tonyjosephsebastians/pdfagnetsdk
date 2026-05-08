from __future__ import annotations

import math
import re
from collections import Counter
from typing import Protocol, Sequence

from pdf_agent_sdk.documents import TextChunk

TOKEN_RE = re.compile(r"[A-Za-z0-9_']+")
STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "how",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "this",
    "to",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
    "with",
}


class Retriever(Protocol):
    """Retrieval strategy used by PDFAgent."""

    def retrieve(self, query: str, chunks: Sequence[TextChunk], *, limit: int = 5) -> list[TextChunk]:
        """Return the chunks most relevant to query."""


class KeywordRetriever:
    """Simple TF-IDF-like keyword retrieval strategy."""

    def retrieve(self, query: str, chunks: Sequence[TextChunk], *, limit: int = 5) -> list[TextChunk]:
        return rank_chunks(query, chunks, limit=limit)


def rank_chunks(query: str, chunks: Sequence[TextChunk], *, limit: int = 5) -> list[TextChunk]:
    if limit <= 0:
        raise ValueError("limit must be positive.")
    if not chunks:
        return []

    query_terms = _tokens(query)
    if not query_terms:
        return list(chunks[:limit])

    document_frequency = Counter()
    chunk_terms = []
    for chunk in chunks:
        terms = _tokens(chunk.text)
        chunk_terms.append(terms)
        document_frequency.update(set(terms))

    scored: list[tuple[float, int, TextChunk]] = []
    total_chunks = len(chunks)
    for index, (chunk, terms) in enumerate(zip(chunks, chunk_terms)):
        if not terms:
            continue
        term_counts = Counter(terms)
        score = 0.0
        for term in set(query_terms):
            if term not in term_counts:
                continue
            inverse_document_frequency = math.log((1 + total_chunks) / (1 + document_frequency[term])) + 1
            score += term_counts[term] * inverse_document_frequency

        if score:
            score = score / math.sqrt(len(terms))
            if query.lower() in chunk.text.lower():
                score += 2.0
            scored.append((score, index, chunk))

    if not scored:
        return list(chunks[:limit])

    scored.sort(key=lambda item: (-item[0], item[1]))
    return [chunk for _, _, chunk in scored[:limit]]


def _tokens(text: str) -> list[str]:
    return [
        token
        for token in (match.group(0).lower() for match in TOKEN_RE.finditer(text))
        if token not in STOPWORDS and len(token) > 1
    ]
