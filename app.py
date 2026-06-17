
"""
Telecom RAG Assistant
A chatbot that answers telecom engineering questions using your own documentation.
 
Setup:
1. Add .txt or .pdf files to the docs/ folder
2. Run: python ingest.py
3. Run: streamlit run app.py
"""
 
import os
import streamlit as st
import chromadb
from chromadb.utils import embedding_functions
from openai import OpenAI
 
# ── Config ──────────────────────────────────────────────────────────────────
CHROMA_DIR = "./chroma_db"
DOCS_DIR = "./docs"
COLLECTION_NAME = "telecom_docs"
TOP_K = 5  # Number of chunks to retrieve per query
 
# DeepSeek API (compatible with OpenAI client)
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEEPSEEK_MODEL = "deepseek-chat"
 
# ── Page Setup ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Telecom RAG Assistant",
    page_icon="📡",
    layout="wide"
)
 
st.title("📡 Telecom RAG Assistant")
st.markdown("Ask questions about telecom engineering — RF optimization, LTE/5G, handovers, KPIs, and more.")
st.divider()
 
# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Configuration")
 
    api_key = st.text_input(
        "DeepSeek API Key",
        value=os.getenv("DEEPSEEK_API_KEY", ""),
        type="password",
        placeholder="sk-...",
        help="Get your key at platform.deepseek.com"
    )
 
    st.divider()
    st.header("📁 Add Documents")
    uploaded_files = st.file_uploader(
        "Upload .txt or .pdf files",
        accept_multiple_files=True,
        type=["txt", "pdf"]
    )
 
    if uploaded_files:
        os.makedirs("docs", exist_ok=True)
        saved = []
        for f in uploaded_files:
            path = os.path.join("docs", f.name)
            with open(path, "wb") as out:
                out.write(f.getbuffer())
            saved.append(f.name)
        st.success(f"Saved: {', '.join(saved)}")
        st.info("Run `python ingest.py` in your terminal to index the new files.")
 
    st.divider()
    st.markdown("**How it works:**")
    st.markdown("1. Your question is converted to a vector embedding")
    st.markdown("2. Most relevant doc chunks are retrieved from ChromaDB")
    st.markdown("3. DeepSeek generates an answer grounded in those chunks")
    st.markdown("4. Sources are shown so you can verify")
 
 
# ── Load Vector Store ─────────────────────────────────────────────────────────
@st.cache_resource
def load_collection():
    try:
        ef = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )
        # Try persistent first (local dev), fall back to in-memory (Streamlit Cloud)
        try:
            client = chromadb.PersistentClient(path=CHROMA_DIR)
            collection = client.get_or_create_collection(
                name=COLLECTION_NAME,
                embedding_function=ef,
                metadata={"hnsw:space": "cosine"}
            )
        except Exception:
            client = chromadb.EphemeralClient()
            collection = client.get_or_create_collection(
                name=COLLECTION_NAME,
                embedding_function=ef,
                metadata={"hnsw:space": "cosine"}
            )

        if collection.count() == 0:
            from ingest import load_documents, chunk_text
            st.info("Building vector store, please wait...")
            documents = load_documents(DOCS_DIR)
            all_chunks, all_ids, all_metadata = [], [], []
            for chunk_id, doc in enumerate([c for d in documents for c in [{"text": ch, "source": d["source"]} for ch in chunk_text(d["text"])]]):
                all_chunks.append(doc["text"])
                all_ids.append(f"chunk_{chunk_id}")
                all_metadata.append({"source": doc["source"]})
            for i in range(0, len(all_chunks), 100):
                collection.add(documents=all_chunks[i:i+100], ids=all_ids[i:i+100], metadatas=all_metadata[i:i+100])

        return collection
    except Exception as e:
        st.error(f"Vector store error: {e}")
        return None
 
 
collection = load_collection()
 
if collection is None:
    st.warning("⚠️ No vector store found. Run `python ingest.py` first to index your documents.")
    st.stop()
 
doc_count = collection.count()
st.caption(f"📚 {doc_count} document chunks indexed and ready.")
 
 
# ── Chat History ──────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
 
# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "sources" in msg:
            with st.expander("📄 Sources"):
                for src in msg["sources"]:
                    st.markdown(f"**{src['source']}**")
                    st.text(src["text"][:300] + "..." if len(src["text"]) > 300 else src["text"])
 
 
# ── Query Function ─────────────────────────────────────────────────────────────
def query_rag(question, api_key):
    # Retrieve relevant chunks
    results = collection.query(
        query_texts=[question],
        n_results=min(TOP_K, doc_count)
    )
 
    chunks = results["documents"][0]
    metadatas = results["metadatas"][0]
 
    context = "\n\n---\n\n".join(chunks)
 
    system_prompt = """You are a senior telecommunications engineer with deep expertise in RF optimization,
LTE/5G networks, CDMA systems, network performance engineering, and multi-vendor environments
(Nokia, Ericsson, Alcatel-Lucent).
 
Answer questions using the provided context from telecom documentation. Be specific and technical.
If the context doesn't contain enough information to fully answer the question, say so clearly
and provide what general expertise you can. Always cite which part of the documentation supports your answer."""
 
    user_prompt = f"""Context from telecom documentation:
{context}
 
Question: {question}
 
Please provide a detailed, technical answer based on the documentation above."""
 
    client = OpenAI(api_key=api_key, base_url=DEEPSEEK_BASE_URL)
    response = client.chat.completions.create(
        model=DEEPSEEK_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.2,
        max_tokens=1000
    )
 
    answer = response.choices[0].message.content
    sources = [{"source": m["source"], "text": t} for m, t in zip(metadatas, chunks)]
 
    return answer, sources
 
 
# ── Chat Input ────────────────────────────────────────────────────────────────
placeholder_questions = [
    "What causes X2 handover failures and how do I fix them?",
    "What RSRP threshold indicates poor LTE coverage?",
    "How do I resolve pilot pollution issues?",
    "What are the key KPIs to monitor for LTE network health?",
    "Explain the difference between NSA and SA 5G architecture.",
]
 
st.markdown("**Try asking:**")
cols = st.columns(len(placeholder_questions))
for i, (col, q) in enumerate(zip(cols, placeholder_questions)):
    if col.button(q[:40] + "...", key=f"suggest_{i}"):
        st.session_state["prefill"] = q
 
if "prefill" in st.session_state:
    question = st.session_state.pop("prefill")
else:
    question = None
 
user_input = st.chat_input("Ask a telecom engineering question...")
if user_input:
    question = user_input
 
if question:
    if not api_key:
        st.error("Please enter your DeepSeek API key in the sidebar.")
        st.stop()
 
    # Show user message
    with st.chat_message("user"):
        st.markdown(question)
    st.session_state.messages.append({"role": "user", "content": question})
 
    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Searching documentation and generating answer..."):
            try:
                answer, sources = query_rag(question, api_key)
                st.markdown(answer)
                with st.expander("📄 Sources used"):
                    for src in sources:
                        st.markdown(f"**{src['source']}**")
                        st.text(src["text"][:300] + "..." if len(src["text"]) > 300 else src["text"])
 
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": sources                })
            except Exception as e:
                st.error(f"Error: {str(e)}")
 
# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.caption("Telecom RAG Assistant | Built with ChromaDB + DeepSeek API + Streamlit | github.com/Joey-Bradley")