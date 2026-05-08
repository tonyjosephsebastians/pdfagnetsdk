from types import SimpleNamespace

from pdf_agent_sdk.providers import AzureOpenAIProvider, OpenAIProvider


class FakeResponses:
    def __init__(self) -> None:
        self.request = None

    def create(self, **kwargs):
        self.request = kwargs
        return SimpleNamespace(output_text="response text")


class FakeChatCompletions:
    def __init__(self) -> None:
        self.request = None

    def create(self, **kwargs):
        self.request = kwargs
        message = SimpleNamespace(content="chat text")
        choice = SimpleNamespace(message=message)
        return SimpleNamespace(choices=[choice])


class FakeClient:
    def __init__(self) -> None:
        self.responses = FakeResponses()
        self.chat = SimpleNamespace(completions=FakeChatCompletions())


def test_openai_provider_uses_responses_api_by_default() -> None:
    client = FakeClient()
    provider = OpenAIProvider(model="test-model", client=client)

    assert provider.complete(system_prompt="sys", user_prompt="user") == "response text"
    assert client.responses.request["model"] == "test-model"
    assert client.responses.request["instructions"] == "sys"


def test_azure_provider_uses_chat_completions_by_default() -> None:
    client = FakeClient()
    provider = AzureOpenAIProvider(
        azure_endpoint="https://example.openai.azure.com",
        azure_deployment="deployment",
        client=client,
    )

    assert provider.complete(system_prompt="sys", user_prompt="user") == "chat text"
    assert client.chat.completions.request["model"] == "deployment"
    assert client.chat.completions.request["messages"][0]["role"] == "system"
