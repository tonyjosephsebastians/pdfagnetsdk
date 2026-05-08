from __future__ import annotations

import os
from typing import Any, Protocol


class LLMProvider(Protocol):
    """Minimal provider interface used by PDFAgent."""

    def complete(self, *, system_prompt: str, user_prompt: str) -> str:
        """Return a text completion for the supplied prompts."""


class OpenAIProvider:
    """OpenAI provider backed by the official openai Python SDK."""

    def __init__(
        self,
        *,
        model: str = "gpt-5.2",
        api_key: str | None = None,
        base_url: str | None = None,
        organization: str | None = None,
        project: str | None = None,
        use_responses_api: bool = True,
        temperature: float | None = None,
        client: Any | None = None,
    ) -> None:
        self.model = model
        self.use_responses_api = use_responses_api
        self.temperature = temperature
        self._client = client or self._build_client(
            api_key=api_key,
            base_url=base_url,
            organization=organization,
            project=project,
        )

    @classmethod
    def from_env(cls) -> "OpenAIProvider":
        return cls(
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("OPENAI_BASE_URL"),
            organization=os.getenv("OPENAI_ORG_ID"),
            project=os.getenv("OPENAI_PROJECT_ID"),
            model=os.getenv("OPENAI_MODEL", "gpt-5.2"),
        )

    def complete(self, *, system_prompt: str, user_prompt: str) -> str:
        if self.use_responses_api:
            request: dict[str, Any] = {
                "model": self.model,
                "instructions": system_prompt,
                "input": user_prompt,
            }
            if self.temperature is not None:
                request["temperature"] = self.temperature
            response = self._client.responses.create(**request)
            text = getattr(response, "output_text", None)
            if text:
                return text
            return _extract_response_text(response)

        request = _chat_request(
            model=self.model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=self.temperature,
        )
        response = self._client.chat.completions.create(**request)
        return response.choices[0].message.content or ""

    def _build_client(
        self,
        *,
        api_key: str | None,
        base_url: str | None,
        organization: str | None,
        project: str | None,
    ) -> Any:
        from openai import OpenAI

        kwargs = _drop_none(
            {
                "api_key": api_key,
                "base_url": base_url,
                "organization": organization,
                "project": project,
            }
        )
        return OpenAI(**kwargs)


class AzureOpenAIProvider:
    """Azure OpenAI provider backed by the official openai Python SDK."""

    def __init__(
        self,
        *,
        azure_endpoint: str,
        azure_deployment: str,
        api_version: str = "2024-02-01",
        api_key: str | None = None,
        azure_ad_token: str | None = None,
        azure_ad_token_provider: Any | None = None,
        use_responses_api: bool = False,
        temperature: float | None = None,
        client: Any | None = None,
    ) -> None:
        self.azure_deployment = azure_deployment
        self.api_version = api_version
        self.use_responses_api = use_responses_api
        self.temperature = temperature
        self._client = client or self._build_client(
            api_key=api_key,
            azure_endpoint=azure_endpoint,
            api_version=api_version,
            azure_ad_token=azure_ad_token,
            azure_ad_token_provider=azure_ad_token_provider,
        )

    @classmethod
    def from_env(cls) -> "AzureOpenAIProvider":
        endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT") or os.getenv("AZURE_OPENAI_MODEL")
        if not endpoint:
            raise ValueError("AZURE_OPENAI_ENDPOINT is required for AzureOpenAIProvider.")
        if not deployment:
            raise ValueError("AZURE_OPENAI_DEPLOYMENT is required for AzureOpenAIProvider.")

        return cls(
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            azure_endpoint=endpoint,
            azure_deployment=deployment,
            api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01"),
        )

    def complete(self, *, system_prompt: str, user_prompt: str) -> str:
        if self.use_responses_api:
            request: dict[str, Any] = {
                "model": self.azure_deployment,
                "instructions": system_prompt,
                "input": user_prompt,
            }
            if self.temperature is not None:
                request["temperature"] = self.temperature
            response = self._client.responses.create(**request)
            text = getattr(response, "output_text", None)
            if text:
                return text
            return _extract_response_text(response)

        request = _chat_request(
            model=self.azure_deployment,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=self.temperature,
        )
        response = self._client.chat.completions.create(**request)
        return response.choices[0].message.content or ""

    def _build_client(
        self,
        *,
        api_key: str | None,
        azure_endpoint: str,
        api_version: str,
        azure_ad_token: str | None,
        azure_ad_token_provider: Any | None,
    ) -> Any:
        from openai import AzureOpenAI

        kwargs = _drop_none(
            {
                "api_key": api_key,
                "azure_endpoint": azure_endpoint,
                "api_version": api_version,
                "azure_ad_token": azure_ad_token,
                "azure_ad_token_provider": azure_ad_token_provider,
            }
        )
        return AzureOpenAI(**kwargs)


class ProviderFactory:
    """Factory for creating provider adapters from configuration."""

    @staticmethod
    def from_env() -> LLMProvider:
        return provider_from_env()

    @staticmethod
    def openai_from_env() -> OpenAIProvider:
        return OpenAIProvider.from_env()

    @staticmethod
    def azure_from_env() -> AzureOpenAIProvider:
        return AzureOpenAIProvider.from_env()

    @staticmethod
    def openai(**kwargs: Any) -> OpenAIProvider:
        return OpenAIProvider(**kwargs)

    @staticmethod
    def azure(**kwargs: Any) -> AzureOpenAIProvider:
        return AzureOpenAIProvider(**kwargs)


def provider_from_env() -> LLMProvider:
    provider = os.getenv("PDF_AGENT_PROVIDER", "openai").strip().lower()
    if provider == "openai":
        return OpenAIProvider.from_env()
    if provider in {"azure", "azure_openai", "azure-openai"}:
        return AzureOpenAIProvider.from_env()
    raise ValueError("PDF_AGENT_PROVIDER must be 'openai' or 'azure'.")


def _chat_request(
    *,
    model: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float | None,
) -> dict[str, Any]:
    request: dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    }
    if temperature is not None:
        request["temperature"] = temperature
    return request


def _extract_response_text(response: Any) -> str:
    output = getattr(response, "output", None) or []
    parts: list[str] = []
    for item in output:
        for content in getattr(item, "content", []) or []:
            text = getattr(content, "text", None)
            if text:
                parts.append(text)
    return "\n".join(parts)


def _drop_none(values: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in values.items() if value is not None}
