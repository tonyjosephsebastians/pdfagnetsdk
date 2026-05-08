from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

from pdf_agent_sdk.documents import PdfPage, TextChunk, chunk_pages, load_pdf
from pdf_agent_sdk.providers import LLMProvider
from pdf_agent_sdk.retrieval import KeywordRetriever, Retriever
from pdf_agent_sdk.types import AgentAnswer


DEFAULT_SYSTEM_PROMPT = """You are a PDF question-answering agent.
Use only the provided PDF context.
Cite supporting chunks with bracketed source numbers like [1].
If the context is insufficient, say what is missing instead of guessing."""


class PDFAgent:
    """In-memory PDF retrieval agent."""

    def __init__(
        self,
        chunks: Sequence[TextChunk],
        provider: LLMProvider,
        *,
        retriever: Retriever | None = None,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    ) -> None:
        if not chunks:
            raise ValueError("PDFAgent requires at least one text chunk.")
        self.chunks = list(chunks)
        self.provider = provider
        self.retriever = retriever or KeywordRetriever()
        self.system_prompt = system_prompt

    @classmethod
    def from_pdf(
        cls,
        path: str | Path,
        provider: LLMProvider,
        *,
        chunk_size: int = 1200,
        overlap: int = 150,
        retriever: Retriever | None = None,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    ) -> "PDFAgent":
        pages = load_pdf(path)
        chunks = chunk_pages(pages, chunk_size=chunk_size, overlap=overlap)
        return cls(chunks, provider, retriever=retriever, system_prompt=system_prompt)

    @classmethod
    def from_pdfs(
        cls,
        paths: Iterable[str | Path],
        provider: LLMProvider,
        *,
        chunk_size: int = 1200,
        overlap: int = 150,
        retriever: Retriever | None = None,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    ) -> "PDFAgent":
        chunks: list[TextChunk] = []
        for path in paths:
            chunks.extend(chunk_pages(load_pdf(path), chunk_size=chunk_size, overlap=overlap))
        return cls(chunks, provider, retriever=retriever, system_prompt=system_prompt)

    @classmethod
    def from_text(
        cls,
        text: str,
        provider: LLMProvider,
        *,
        source: str = "text",
        chunk_size: int = 1200,
        overlap: int = 150,
        retriever: Retriever | None = None,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    ) -> "PDFAgent":
        pages = [PdfPage(page_number=1, text=text, source=source)]
        chunks = chunk_pages(pages, chunk_size=chunk_size, overlap=overlap)
        return cls(chunks, provider, retriever=retriever, system_prompt=system_prompt)

    def ask(self, question: str, *, top_k: int = 5) -> AgentAnswer:
        if not question.strip():
            raise ValueError("question must not be empty.")

        selected = self.retriever.retrieve(question, self.chunks, limit=top_k)
        context = self._format_context(selected)
        user_prompt = f"""Question:
{question}

PDF context:
{context}

Answer with concise reasoning and source citations."""

        answer = self.provider.complete(
            system_prompt=self.system_prompt,
            user_prompt=user_prompt,
        )
        return AgentAnswer(answer=answer, question=question, sources=selected)

    def _format_context(self, chunks: Sequence[TextChunk]) -> str:
        blocks = []
        for idx, chunk in enumerate(chunks, start=1):
            location = f"{chunk.source}, page {chunk.page_start}"
            if chunk.page_end != chunk.page_start:
                location = f"{chunk.source}, pages {chunk.page_start}-{chunk.page_end}"
            blocks.append(f"[{idx}] {location}\n{chunk.text}")
        return "\n\n".join(blocks)
