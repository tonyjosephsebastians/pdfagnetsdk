from pdf_agent_sdk import PDFAgent


class FakeProvider:
    def __init__(self) -> None:
        self.system_prompt = ""
        self.user_prompt = ""

    def complete(self, *, system_prompt: str, user_prompt: str) -> str:
        self.system_prompt = system_prompt
        self.user_prompt = user_prompt
        return "The answer is in the contract [1]."


def test_agent_uses_retrieved_context() -> None:
    provider = FakeProvider()
    agent = PDFAgent.from_text(
        "Payment is due within 30 days.\n\nTermination requires 10 days notice.",
        provider=provider,
        source="contract.pdf",
        chunk_size=200,
    )

    result = agent.ask("When is payment due?", top_k=1)

    assert result.answer == "The answer is in the contract [1]."
    assert len(result.sources) == 1
    assert "Payment is due" in provider.user_prompt
    assert "contract.pdf" in provider.user_prompt
