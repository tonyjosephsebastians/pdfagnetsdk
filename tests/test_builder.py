import pytest

from pdf_agent_sdk import PDFAgentBuilder


class FakeProvider:
    def complete(self, *, system_prompt: str, user_prompt: str) -> str:
        return "ok"


def test_builder_creates_agent_from_text() -> None:
    agent = (
        PDFAgentBuilder()
        .with_provider(FakeProvider())
        .add_text("Invoices are due within 30 days.", source="terms.txt")
        .with_chunking(chunk_size=200, overlap=0)
        .build()
    )

    result = agent.ask("When are invoices due?")

    assert result.answer == "ok"
    assert result.sources[0].source == "terms.txt"


def test_builder_requires_source() -> None:
    with pytest.raises(ValueError):
        PDFAgentBuilder().with_provider(FakeProvider()).build()
