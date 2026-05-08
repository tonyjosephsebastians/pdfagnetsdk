from __future__ import annotations

from dataclasses import dataclass

from pdf_agent_sdk.documents import TextChunk


@dataclass(frozen=True)
class AgentAnswer:
    answer: str
    question: str
    sources: list[TextChunk]
