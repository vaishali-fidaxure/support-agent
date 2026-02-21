"""
RAG-based Support Assistant - Streamlit app.
Uses local embeddings (sentence-transformers) and Ollama for the LLM.
"""

from pathlib import Path
import os
import streamlit as st
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# Configuration - must match ingest_documents.py
CHROMA_DIR = Path("./chroma_db")
# 2. Trigger ingestion if the folder is missing
if not os.path.exists(CHROMA_DIR):
    st.warning("Vector database not found. Initializing indexing...")
    try:
        run_ingestion()
        st.success("Database indexed successfully!")
        # Optional: Force a rerun to refresh the state
        st.rerun()
    except Exception as e:
        st.error(f"Error during ingestion: {e}")
        st.stop() # Stop the app if it can't index
COLLECTION_NAME = "support_docs"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "gemma3:1b"  # use: ollama list to see installed models; ollama pull llama2 to add more

RAG_PROMPT = """You are a helpful customer support assistant. Answer the question based only on the following context from company policy documents. If the answer cannot be found in the context, say so. Be concise and accurate.

Context:
{context}

Question: {question}

Answer:"""


def load_retriever():
    """Load the ChromaDB vector store and return a retriever."""
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    vectordb = Chroma(
        persist_directory=str(CHROMA_DIR),
        embedding_function=embeddings,
        collection_name=COLLECTION_NAME,
    )
    return vectordb.as_retriever(search_kwargs={"k": 4})


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def main():
    st.set_page_config(page_title="Support Assistant", page_icon="💬")
    st.title("💬 Support Assistant")
    st.caption("Ask questions about return policy, warranty, and shipping. Powered by local AI.")

    if not CHROMA_DIR.exists():
        st.error(
            f"Vector store not found at `{CHROMA_DIR}`. Run `python ingest.py` first to index your documents."
        )
        return

    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Load retriever and LLM (cached)
    @st.cache_resource
    def get_chain():
        retriever = load_retriever()
        llm = ChatOllama(
            model=LLM_MODEL,
            temperature=0,
            base_url="http://127.0.0.1:11434",
        )
        prompt = ChatPromptTemplate.from_template(RAG_PROMPT)

        chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )
        return chain

    try:
        chain = get_chain()
    except Exception as e:
        st.error(
            f"Failed to load Ollama. Make sure Ollama is running and you've pulled a model: `ollama pull {LLM_MODEL}`\n\nError: {e}"
        )
        return

    # Display chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input("Ask about returns, warranty, or shipping..."):
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    response = chain.invoke(prompt)
                    st.markdown(response)
                except Exception as e:
                    st.error(f"Error: {e}")
                    response = str(e)

        st.session_state.messages.append({"role": "assistant", "content": response})


if __name__ == "__main__":
    main()


