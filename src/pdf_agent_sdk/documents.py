from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, Sequence


class PDFLoadError(RuntimeError):
    """Raised when a PDF cannot be loaded or contains no extractable text."""


@dataclass(frozen=True)
class PdfPage:
    page_number: int
    text: str
    source: str


@dataclass(frozen=True)
class TextChunk:
    id: str
    text: str
    source: str
    page_start: int
    page_end: int
    metadata: dict[str, Any] = field(default_factory=dict)


class PDFLoader(Protocol):
    """PDF loading strategy."""

    def load(self, path: str | Path) -> list[PdfPage]:
        """Load text pages from a PDF path."""


class PypdfPDFLoader:
    """PDF loader adapter backed by pypdf."""

    def load(self, path: str | Path) -> list[PdfPage]:
        return load_pdf(path)


def load_pdf(path: str | Path) -> list[PdfPage]:
    pdf_path = Path(path)
    if not pdf_path.exists():
        raise PDFLoadError(f"PDF does not exist: {pdf_path}")
    if not pdf_path.is_file():
        raise PDFLoadError(f"PDF path is not a file: {pdf_path}")

    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise PDFLoadError("Install pypdf to load PDFs: pip install pypdf") from exc

    try:
        reader = PdfReader(str(pdf_path))
    except Exception as exc:
        raise PDFLoadError(f"Unable to read PDF: {pdf_path}") from exc

    pages: list[PdfPage] = []
    for index, page in enumerate(reader.pages, start=1):
        text = normalize_text(page.extract_text() or "")
        if text:
            pages.append(PdfPage(page_number=index, text=text, source=str(pdf_path)))

    if not pages:
        raise PDFLoadError(f"No extractable text found in PDF: {pdf_path}")
    return pages


def chunk_pages(
    pages: Sequence[PdfPage],
    *,
    chunk_size: int = 1200,
    overlap: int = 150,
) -> list[TextChunk]:
    if chunk_size < 200:
        raise ValueError("chunk_size must be at least 200 characters.")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be non-negative and smaller than chunk_size.")

    chunks: list[TextChunk] = []
    for page in pages:
        text = normalize_text(page.text)
        if not text:
            continue
        for idx, chunk_text in enumerate(_split_text(text, chunk_size=chunk_size, overlap=overlap)):
            chunks.append(
                TextChunk(
                    id=f"{page.source}:p{page.page_number}:c{idx + 1}",
                    text=chunk_text,
                    source=page.source,
                    page_start=page.page_number,
                    page_end=page.page_number,
                    metadata={"chunk_index": idx + 1},
                )
            )
    return chunks


def normalize_text(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t\r\f\v]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _split_text(text: str, *, chunk_size: int, overlap: int) -> list[str]:
    if len(text) <= chunk_size:
        return [text]

    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        if end < len(text):
            boundary = max(text.rfind("\n\n", start, end), text.rfind(". ", start, end))
            if boundary > start + chunk_size // 2:
                end = boundary + 1

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(text):
            break
        start = max(0, end - overlap)

    return chunks
