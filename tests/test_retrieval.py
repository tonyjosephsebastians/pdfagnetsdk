from pdf_agent_sdk.documents import TextChunk
from pdf_agent_sdk.retrieval import rank_chunks


def test_rank_chunks_prefers_matching_terms() -> None:
    chunks = [
        TextChunk(id="1", source="a.pdf", page_start=1, page_end=1, text="Confidentiality lasts two years."),
        TextChunk(id="2", source="a.pdf", page_start=2, page_end=2, text="Invoices are due within 30 days."),
    ]

    ranked = rank_chunks("When are invoices due?", chunks, limit=1)

    assert ranked[0].id == "2"
