# AI-Assistant

A RAG-based support assistant with Streamlit UI. Uses **local models** for embeddings (sentence-transformers) and LLM (Ollama)—no API keys required.

## Prerequisites

- **Ollama** — Install from [ollama.ai](https://ollama.ai), then pull a model:
  ```powershell
  ollama pull llama2
  ```
  (Or use `mistral`, `phi`, etc.)

## Setup

1. Create and activate a virtual environment:

   ```powershell
   py -3.13 -m venv venv
   .\venv\Scripts\Activate.ps1
   ```

2. Install dependencies:

   ```powershell
   pip install -r requirements.txt
   ```

3. Place PDF documents in the `documents/` folder.

## Usage

### 1. Ingest documents (build the knowledge base)

```powershell
python ingest.py
```

This loads PDFs, creates embeddings with `sentence-transformers/all-MiniLM-L6-v2`, and stores them in ChromaDB. **No API key needed.**

### 2. Run the support assistant

```powershell
streamlit run app.py
```

Make sure Ollama is running in the background before starting the app.

## Project Structure

```
AI-Assistant/
├── app.py              # Streamlit RAG chat app
├── ingest.py           # Document ingestion script
├── ingest_documents.py # Ingestion logic
├── documents/          # PDF documents for the knowledge base
├── chroma_db/          # Vector store (created by ingest)
├── requirements.txt
└── README.md
```
