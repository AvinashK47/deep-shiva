import os
from config import settings

from llama_index.core import Settings as LlamaSettings
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama


def configure_llamaindex() -> None:
    provider = settings.provider

    # Embeddings
    if provider == "ollama":
        base_url = os.getenv("OLLAMA_HOST", "http://localhost:11434")
        timeout = float(os.getenv("OLLAMA_REQUEST_TIMEOUT", "180"))
        embed_model = OllamaEmbedding(
            model_name=settings.ollama_embed_model,
            base_url=base_url,
            request_timeout=timeout,
        )
    else:
        # default to openai embeddings
        if not settings.openai_api_key:
            raise RuntimeError("OPENAI_API_KEY not set. Set PROVIDER=ollama to avoid OpenAI.")
        embed_model = OpenAIEmbedding(model=settings.openai_embed_model, api_key=settings.openai_api_key)

    # LLM
    if provider == "ollama":
        base_url = os.getenv("OLLAMA_HOST", "http://localhost:11434")
        timeout = float(os.getenv("OLLAMA_REQUEST_TIMEOUT", "180"))
        num_ctx = int(os.getenv("OLLAMA_NUM_CTX", "4096"))
        temperature = float(os.getenv("OLLAMA_TEMPERATURE", "0.2"))
        llm = Ollama(
            model=settings.ollama_model,
            base_url=base_url,
            request_timeout=timeout,
            context_window=num_ctx,
            temperature=temperature,
        )
    else:
        if not settings.openai_api_key:
            raise RuntimeError("OPENAI_API_KEY not set for OpenAI provider. Set PROVIDER=ollama to use Ollama.")
        llm = OpenAI(model=settings.openai_model, api_key=settings.openai_api_key)

    LlamaSettings.embed_model = embed_model
    LlamaSettings.llm = llm
