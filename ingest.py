"""
ingest.py — Load telecom documents into Supabase pgvector

Run once before launching the app (or whenever you add new docs):
  python ingest.py

Requires environment variables (set in .env):
  SUPABASE_URL          — https://your-project.supabase.co
  SUPABASE_SERVICE_KEY  — service role key (bypasses RLS for writes)
"""

import os
import glob
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from supabase import create_client

load_dotenv()

DOCS_DIR = "./docs"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # 384-dim, runs locally
BATCH_SIZE = 50


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split text into overlapping chunks."""
    chunks = []
    start = 0
    while start < len(text):
        chunk = text[start : start + chunk_size]
        if chunk.strip():
            chunks.append(chunk)
        start += chunk_size - overlap
    return chunks


def load_documents(docs_dir: str) -> list[dict]:
    """Load .txt and .pdf files from the docs directory."""
    documents = []

    for filepath in glob.glob(os.path.join(docs_dir, "*.txt")):
        with open(filepath, "r", encoding="utf-8") as f:
            text = f.read()
        documents.append({"text": text, "source": os.path.basename(filepath)})
        print(f"  Loaded: {filepath}")

    try:
        from pypdf import PdfReader

        for filepath in glob.glob(os.path.join(docs_dir, "*.pdf")):
            reader = PdfReader(filepath)
            text = "\n".join(page.extract_text() or "" for page in reader.pages)
            documents.append({"text": text, "source": os.path.basename(filepath)})
            print(f"  Loaded PDF: {filepath}")
    except ImportError:
        print("  pypdf not installed — skipping PDF files.")

    return documents


def ingest():
    print("=" * 50)
    print("Telecom RAG — Document Ingestion")
    print("=" * 50)

    # Connect to Supabase with the service role key (bypasses RLS)
    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_SERVICE_KEY")

    if not url or not key:
        raise ValueError(
            "SUPABASE_URL and SUPABASE_SERVICE_KEY must be set in .env\n"
            "Find your service role key in Supabase → Project Settings → API"
        )

    supabase = create_client(url, key)
    print("Connected to Supabase.")

    # Load embedding model
    print(f"\nLoading embedding model: {EMBEDDING_MODEL}")
    model = SentenceTransformer(EMBEDDING_MODEL)

    # Load and chunk documents
    print(f"\nScanning docs/ directory...")
    documents = load_documents(DOCS_DIR)
    if not documents:
        print(f"No documents found in {DOCS_DIR}/")
        print("Add .txt or .pdf files and re-run.")
        return

    print(f"\nFound {len(documents)} document(s). Chunking...")

    # Clear existing data
    supabase.table("telecom_docs").delete().neq("id", 0).execute()
    print("Cleared existing records.")

    # Build rows
    all_rows = []
    for doc in documents:
        chunks = chunk_text(doc["text"])
        for chunk in chunks:
            embedding = model.encode(chunk).tolist()
            all_rows.append(
                {"content": chunk, "source": doc["source"], "embedding": embedding}
            )

    print(f"\nInserting {len(all_rows)} chunks in batches of {BATCH_SIZE}...")
    for i in range(0, len(all_rows), BATCH_SIZE):
        batch = all_rows[i : i + BATCH_SIZE]
        supabase.table("telecom_docs").insert(batch).execute()
        end = min(i + BATCH_SIZE, len(all_rows))
        print(f"  Inserted chunks {i + 1}–{end}")

    print("\n" + "=" * 50)
    print(f"Ingestion complete!")
    print(f"  Documents : {len(documents)}")
    print(f"  Chunks    : {len(all_rows)}")
    print("=" * 50)
    print("\nYou can now run: streamlit run app.py")


if __name__ == "__main__":
    ingest()
