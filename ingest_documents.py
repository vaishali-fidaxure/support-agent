"""
Document ingestion script for RAG support assistant.
Loads PDFs from ./documents/, chunks them, creates embeddings, and stores in ChromaDB.
Uses local sentence-transformers for embeddings (no API key required).
"""

from pathlib import Path

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Load environment variables from .env
load_dotenv()

# Configuration
DOCUMENTS_DIR = Path("./documents")
CHROMA_DIR = Path("./chroma_db")
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
COLLECTION_NAME = "support_docs"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def main():
    if not DOCUMENTS_DIR.exists():
        print(f"Error: Documents directory '{DOCUMENTS_DIR}' not found.")
        return

    pdf_files = sorted(DOCUMENTS_DIR.glob("*.pdf"))
    if not pdf_files:
        print(f"No PDF files found in '{DOCUMENTS_DIR}'.")
        return

    print(f"Found {len(pdf_files)} PDF(s) to process.\n")

    # 1. Load all PDFs using PyPDFLoader
    print("Loading PDFs...")
    documents = []
    for i, pdf_path in enumerate(pdf_files, 1):
        print(f"  [{i}/{len(pdf_files)}] {pdf_path.name}")
        loader = PyPDFLoader(str(pdf_path))
        docs = loader.load()
        documents.extend(docs)
    total_pages = len(documents)
    print(f"  Loaded {total_pages} page(s) from {len(pdf_files)} file(s).\n")

    # 2. Split into chunks
    print("Splitting into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
    )
    chunks = text_splitter.split_documents(documents)
    print(f"  Created {len(chunks)} chunk(s).\n")

    # 3. Create embeddings and store in ChromaDB (local model, no API key needed)
    print("Creating embeddings and storing in ChromaDB...")
    print(f"  Using model: {EMBEDDING_MODEL}")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    # Use persist_directory so data is saved to disk
    Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=str(CHROMA_DIR),
        collection_name=COLLECTION_NAME,
    )
    print("  Done.\n")

    # 4. Print stats
    print("=" * 50)
    print("INGESTION COMPLETE")
    print("=" * 50)
    print(f"  PDF files processed: {len(pdf_files)}")
    print(f"  Total pages:         {total_pages}")
    print(f"  Chunks created:      {len(chunks)}")
    print(f"  Chunk size:          {CHUNK_SIZE} chars")
    print(f"  Chunk overlap:       {CHUNK_OVERLAP} chars")
    print(f"  Vector store:        {CHROMA_DIR.absolute()}")
    print("=" * 50)


if __name__ == "__main__":
    main()
