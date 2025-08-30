from dataclasses import dataclass
import os


@dataclass
class Settings:
    provider: str = os.getenv("PROVIDER", "openai").lower()
    openai_api_key: str | None = os.getenv("OPENAI_API_KEY")
    openai_model: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    openai_embed_model: str = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")

    ollama_model: str = os.getenv("OLLAMA_MODEL", "gemma3:4b")
    ollama_embed_model: str = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")

    chroma_path: str = os.getenv("CHROMA_PATH", os.path.join("storage", "chroma"))
    data_dir: str = os.getenv("DATA_DIR", "data")
    index_name: str = os.getenv("INDEX_NAME", "default")


settings = Settings()
