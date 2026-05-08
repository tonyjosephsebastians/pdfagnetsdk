import pytest

from pdf_agent_sdk.documents import PdfPage, chunk_pages


def test_chunk_pages_splits_long_text_with_overlap() -> None:
    text = "A" * 250 + ". " + "B" * 250 + ". " + "C" * 250
    pages = [PdfPage(page_number=3, text=text, source="sample.pdf")]

    chunks = chunk_pages(pages, chunk_size=300, overlap=50)

    assert len(chunks) >= 3
    assert chunks[0].source == "sample.pdf"
    assert chunks[0].page_start == 3
    assert chunks[0].id == "sample.pdf:p3:c1"


def test_chunk_pages_validates_settings() -> None:
    pages = [PdfPage(page_number=1, text="hello", source="sample.pdf")]

    with pytest.raises(ValueError):
        chunk_pages(pages, chunk_size=100)

    with pytest.raises(ValueError):
        chunk_pages(pages, chunk_size=200, overlap=200)
