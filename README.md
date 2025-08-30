# Deep Shiva Tourism Chatbot

CURRENTLY terminal based RAG chatbot using LlamaIndex + ChromaDB with OpenAI (default) and optional Ollama (gemma3:4b). 

Drop documents/CSVs into `ai/data/` and run ingestion - it incrementally updates the vector DB.

## Use
- Ask about tourism statistics
- Ask about the weather in any Uttarakhand city
- Ask about the Bhagvad Gita
- And more!


## Quick start
1. cd `apps/ai`
2. Create a virtual environment (`python -m venv .venv`) and install deps (`pip install -r requirements.txt`).
3. Copy `.env.template` to `.env` and fill values.
4. Ingest data: `python ingest.py`
5. Chat: `python chat.py`


## Structure
- `apps/ai/data/` - your source docs (md, txt, pdf, docx, csv, json, …)
- `apps/ai/storage/` - ChromaDB persistence and small state for incremental updates
- `apps/ai/rag/` - ingestion and chat CLI


Backend and Frontend are currently works in progress and will be under `apps/backend` and `apps/frontend/` respectively.
