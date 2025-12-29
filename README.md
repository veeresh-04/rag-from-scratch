# RAG From Scratch

An end-to-end Retrieval Augmented Generation (RAG) system built from scratch
using semantic embeddings and a locally hosted LLM via Ollama.

## Features
- Document ingestion and chunking
- Semantic search using cosine similarity
- Local LLM inference (Ollama)
- Grounded answers with source attribution
- Interactive CLI

Requires:
- Ollama installed
- Model: llama3.2 pulled locally


## How to Run
```bash
python ingest.py
python embed.py
python app.py
