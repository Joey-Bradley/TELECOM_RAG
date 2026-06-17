"""
ingest.py - Load telecom documents into ChromaDB vector store
Run this once before launching the app: python ingest.py
"""

import os
import glob
import chromadb
from chromadb.utils import embedding_functions

DOCS_DIR = "./docs"
CHROMA_DIR = "./chroma_db"
COLLECTION_NAME = "telecom_docs"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50


def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """Split text into overlapping chunks."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        if chunk.strip():
            chunks.append(chunk)
        start += chunk_size - overlap
    return chunks


def load_documents(docs_dir):
    """Load all .txt and .pdf files from the docs directory."""
    documents = []

    # Load .txt files
    for filepath in glob.glob(os.path.join(docs_dir, "*.txt")):
        with open(filepath, "r", encoding="utf-8") as f:
            text = f.read()
        documents.append({"text": text, "source": os.path.basename(filepath)})
        print(f"Loaded: {filepath}")

    # Load .pdf files if pypdf is available
    try:
        from pypdf import PdfReader
        for filepath in glob.glob(os.path.join(docs_dir, "*.pdf")):
            reader = PdfReader(filepath)
            text = ""
            for page in reader.pages:
                text += page.extract_text() + "\n"
            documents.append({"text": text, "source": os.path.basename(filepath)})
            print(f"Loaded PDF: {filepath}")
    except ImportError:
        pass

    return documents


def ingest():
    print("Starting document ingestion...")

    # Set up embedding function using sentence-transformers (free, local)
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )

    # Initialize ChromaDB
    client = chromadb.PersistentClient(path=CHROMA_DIR)

    # Delete existing collection if it exists (fresh ingest)
    try:
        client.delete_collection(COLLECTION_NAME)
        print("Cleared existing collection.")
    except Exception:
        pass

    collection = client.create_collection(
        name=COLLECTION_NAME,
        embedding_function=ef,
        metadata={"hnsw:space": "cosine"}
    )

    # Load and chunk documents
    documents = load_documents(DOCS_DIR)
    if not documents:
        print(f"No documents found in {DOCS_DIR}/")
        print("Add .txt or .pdf files to the docs/ folder and run again.")
        return

    all_chunks = []
    all_ids = []
    all_metadata = []

    chunk_id = 0
    for doc in documents:
        chunks = chunk_text(doc["text"])
        for chunk in chunks:
            all_chunks.append(chunk)
            all_ids.append(f"chunk_{chunk_id}")
            all_metadata.append({"source": doc["source"]})
            chunk_id += 1

    # Add to ChromaDB in batches
    batch_size = 100
    for i in range(0, len(all_chunks), batch_size):
        collection.add(
            documents=all_chunks[i:i+batch_size],
            ids=all_ids[i:i+batch_size],
            metadatas=all_metadata[i:i+batch_size]
        )

    print(f"\nIngestion complete!")
    print(f"  Documents loaded: {len(documents)}")
    print(f"  Total chunks stored: {len(all_chunks)}")
    print(f"  Vector store saved to: {CHROMA_DIR}/")
    print("\nYou can now run the app: streamlit run app.py")


if __name__ == "__main__":
    ingest()
