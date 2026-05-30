# PDF Agent SDK

A small Python SDK for asking questions over PDFs with either OpenAI or Azure OpenAI.

The SDK extracts PDF text, chunks it, retrieves the most relevant chunks for a question, and sends grounded context to an LLM provider. 

## Install

```bash
pip install -e .
```

## OpenAI

```powershell
$env:OPENAI_API_KEY="..."
$env:OPENAI_MODEL="gpt-5.2"
```

```python
from pdf_agent_sdk import OpenAIProvider, PDFAgent

agent = PDFAgent.from_pdf(
    "sample.pdf",
    provider=OpenAIProvider.from_env(),
)

result = agent.ask("What are the key obligations in this document?")
print(result.answer)
```

## Azure OpenAI

For Azure OpenAI, `model` is your name.

```powershell
$env:AZURE_OPENAI_API_KEY="..."
$env:AZURE_OPENAI_ENDPOINT="https://YOUR-RESOURCE.openai.azure.com"
$env:AZURE_OPENAI_API_VERSION="2024-02-01"
$env:AZURE_OPENAI_DEPLOYMENT="YOUR-DEPLOYMENT"
```

```python
from pdf_agent_sdk import AzureOpenAIProvider, PDFAgent

agent = PDFAgent.from_pdf(
    "sample.pdf",
    provider=AzureOpenAIProvider.from_env(),
)

result = agent.ask("Summarize the termination clause.")
print(result.answer)
```

## Provider Factory

Set `PDF_AGENT_PROVIDER=openai` or `PDF_AGENT_PROVIDER=azure`.

```python
from pdf_agent_sdk import PDFAgent, provider_from_env

agent = PDFAgent.from_pdf("sample.pdf", provider=provider_from_env())
print(agent.ask("What is this PDF about?").answer)
```

## Builder API

Use the builder when your application needs to assemble an agent from multiple sources  custom strategies.

```python
from pdf_agent_sdk import PDFAgentBuilder

agent = (
    PDFAgentBuilder()
    .with_provider(ProviderFactory.openai(model="gpt-5.2"))
    .add_pdf("contract.pdf")
    .add_pdf("appendix.pdf")
    .with_chunking(chunk_size=1000, overlap=100)
    .build()
)

print(agent.ask("What are the renewal terms?").answer)
```

## Design Patterns

- Facade: `PDFAgent` gives applications one simple API: `ask(...)`.
- Strategy: `LLMProvider` and `Retriever` let you swap model providers and retrieval behavior.
- Adapter: `OpenAIProvider` and `AzureOpenAIProvider` wrap the official OpenAI client behind the same SDK contract.
- Factory: `ProviderFactory` and `provider_from_env()` centralize provider construction.
- Builder: `PDFAgentBuilder`  agents from PDFs, text, chunking settings, provider adapters, and retrieval strategies.

## Notes

- OpenAI defaults to the Responses API.
- Azure OpenAI defaults to Chat Completions for broad deployment compatibility.
- The SDK only answers with retrieved PDF context and asks the model to cite source chunk numbers.
