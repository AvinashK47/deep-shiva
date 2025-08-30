# Deep Shiva Tourism Chatbot

Backend first RAG chatbot using LlamaIndex + ChromaDB with OpenAI (default) and optional Ollama (gemma3:4b). 

Drop documents/CSVs into `ai/data/` and run ingestion - it incrementally updates the vector DB.

## Quick start

1. cd `ai`
2. Create a virtual environment (`python -m venv .venv`) and install deps (`pip install -r requirements.txt`).
3. Copy `.env.template` to `.env` and fill values.
4. Ingest data: `python ingest.py`
5. Chat: `python chat.py`

## Structure
- `ai/data/` - your source docs (md, txt, pdf, docx, csv, json, …)
- `ai/storage/` - ChromaDB persistence and small state for incremental updates
- `ai/rag/` - ingestion and chat CLI

Frontend will be added later under `frontend/`.
