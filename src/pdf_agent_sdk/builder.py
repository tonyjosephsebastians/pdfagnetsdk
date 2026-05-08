from __future__ import annotations

from pathlib import Path

from pdf_agent_sdk.agent import DEFAULT_SYSTEM_PROMPT, PDFAgent
from pdf_agent_sdk.documents import PdfPage, chunk_pages, load_pdf
from pdf_agent_sdk.providers import LLMProvider, ProviderFactory
from pdf_agent_sdk.retrieval import Retriever


class PDFAgentBuilder:
    """Fluent builder for composing a PDFAgent."""

    def __init__(self) -> None:
        self._provider: LLMProvider | None = None
        self._pdf_paths: list[str | Path] = []
        self._text_sources: list[tuple[str, str]] = []
        self._chunk_size = 1200
        self._overlap = 150
        self._retriever: Retriever | None = None
        self._system_prompt = DEFAULT_SYSTEM_PROMPT

    def with_provider(self, provider: LLMProvider) -> "PDFAgentBuilder":
        self._provider = provider
        return self

    def with_openai_from_env(self) -> "PDFAgentBuilder":
        self._provider = ProviderFactory.openai_from_env()
        return self

    def with_azure_openai_from_env(self) -> "PDFAgentBuilder":
        self._provider = ProviderFactory.azure_from_env()
        return self

    def add_pdf(self, path: str | Path) -> "PDFAgentBuilder":
        self._pdf_paths.append(path)
        return self

    def add_text(self, text: str, *, source: str = "text") -> "PDFAgentBuilder":
        self._text_sources.append((text, source))
        return self

    def with_chunking(self, *, chunk_size: int = 1200, overlap: int = 150) -> "PDFAgentBuilder":
        self._chunk_size = chunk_size
        self._overlap = overlap
        return self

    def with_retriever(self, retriever: Retriever) -> "PDFAgentBuilder":
        self._retriever = retriever
        return self

    def with_system_prompt(self, system_prompt: str) -> "PDFAgentBuilder":
        self._system_prompt = system_prompt
        return self

    def build(self) -> PDFAgent:
        provider = self._provider or ProviderFactory.from_env()
        pages: list[PdfPage] = []
        for path in self._pdf_paths:
            pages.extend(load_pdf(path))
        for index, (text, source) in enumerate(self._text_sources, start=1):
            pages.append(PdfPage(page_number=index, text=text, source=source))

        if not pages:
            raise ValueError("PDFAgentBuilder requires at least one PDF or text source.")

        chunks = chunk_pages(pages, chunk_size=self._chunk_size, overlap=self._overlap)
        return PDFAgent(
            chunks,
            provider,
            retriever=self._retriever,
            system_prompt=self._system_prompt,
        )
