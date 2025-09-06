# Deep Shiva Tourism Chatbot

CURRENTLY terminal based RAG chatbot using LlamaIndex + ChromaDB with OpenAI (default) and optional Ollama (gemma3:4b).

You can mix providers: use OpenAI embeddings with an Ollama LLM (or vice versa). Switch LLMs without re-ingesting.

Drop documents/CSVs/etc into `apps/ai/data/` and run ingestion - it incrementally updates the vector DB.

## Use
- Ask about tourism statistics
- Ask about the weather in any Uttarakhand city
- Ask about the Bhagvad Gita
- And more!


## Quick start
1. cd `apps/ai`
2. Create a virtual environment (`python -m venv .venv`)
3. Activate venv (`.venv/scripts/activate`)
4. Install dependencies (`pip install -r requirements.txt`)
5. Set llm_provider and embed_provider in `config.yaml`
6. Copy `.env.template` to `.env` and fill values. 
7. Ingest data: `python ingest.py`
8. Chat: `python chat.py`


## Structure
- `apps/ai/data/` - your source docs (md, txt, pdf, docx, csv, json, …)
- `apps/ai/storage/` - ChromaDB persistence and small state for incremental updates
- `apps/ai/rag/` - ingestion and chat CLI

Frontend is currently a work in progress and will be under `apps/frontend/` shortly.
