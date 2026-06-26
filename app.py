"""
Telecom RAG Assistant
A chatbot that answers telecom engineering questions using your own documentation.

Environment variables (set in .env locally or Streamlit Cloud secrets):
  SUPABASE_URL
  SUPABASE_KEY
  DEEPSEEK_API_KEY  (optional — users can also enter their own in the sidebar)
"""

import os
import streamlit as st
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from supabase import create_client
from openai import OpenAI

load_dotenv()

# ── Config ────────────────────────────────────────────────────────────────────
TOP_K = 5
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEEPSEEK_MODEL = "deepseek-chat"

st.set_page_config(
    page_title="Telecom RAG Assistant",
    page_icon="📡",
    layout="wide",
)

st.title("📡 Telecom RAG Assistant")
st.markdown("Ask questions about telecom engineering — RF optimization, LTE/5G, handovers, KPIs, and more.")
st.divider()


# ── Helpers ───────────────────────────────────────────────────────────────────
def get_secret(key: str, default: str = "") -> str:
    """Read from environment variables or Streamlit secrets."""
    val = os.environ.get(key)
    if val:
        return val
    try:
        return st.secrets.get(key, default)
    except Exception:
        return default


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Configuration")

    server_api_key = get_secret("DEEPSEEK_API_KEY")
    if server_api_key:
        api_key = server_api_key
        st.success("API key configured ✓")
    else:
        api_key = st.text_input(
            "DeepSeek API Key",
            type="password",
            placeholder="sk-...",
            help="Get your key at platform.deepseek.com",
        )

    st.divider()
    st.markdown("**How it works:**")
    st.markdown("1. Your question is embedded locally (sentence-transformers)")
    st.markdown("2. Most relevant chunks are retrieved from Supabase pgvector")
    st.markdown("3. DeepSeek generates an answer grounded in those chunks")
    st.markdown("4. Sources are shown so you can verify")


# ── Load Resources ────────────────────────────────────────────────────────────
@st.cache_resource
def load_resources():
    url = get_secret("SUPABASE_URL")
    key = get_secret("SUPABASE_KEY")
    if not url or not key:
        return None, None
    supabase = create_client(url, key)
    model = SentenceTransformer(EMBEDDING_MODEL)
    return supabase, model


supabase, embed_model = load_resources()

if supabase is None:
    st.error("⚠️ Supabase credentials not found. Set SUPABASE_URL and SUPABASE_KEY in Streamlit secrets.")
    st.stop()

try:
    count_resp = supabase.table("telecom_docs").select("id", count="exact").execute()
    doc_count = count_resp.count or 0
except Exception as e:
    st.error(f"Could not connect to Supabase: {e}")
    st.stop()

if doc_count == 0:
    st.warning("⚠️ No documents indexed. Run `python ingest.py` to load your docs.")
    st.stop()

st.caption(f"📚 {doc_count} document chunks indexed and ready.")


# ── Chat History ──────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "sources" in msg:
            with st.expander("📄 Sources"):
                for src in msg["sources"]:
                    st.markdown(f"**{src['source']}**")
                    st.text(src["text"][:300] + "..." if len(src["text"]) > 300 else src["text"])


# ── Query Function ────────────────────────────────────────────────────────────
def query_rag(question: str, key: str):
    query_embedding = embed_model.encode(question).tolist()

    response = supabase.rpc("match_telecom_docs", {
        "query_embedding": query_embedding,
        "match_count": TOP_K,
    }).execute()

    results = response.data or []
    if not results:
        return "No relevant documentation found for that question.", []

    sources = [{"source": r["source"], "text": r["content"]} for r in results]
    context = "\n\n---\n\n".join(r["content"] for r in results)

    system_prompt = (
        "You are a senior telecommunications engineer with deep expertise in RF optimization, "
        "LTE/5G networks, CDMA systems, network performance engineering, and multi-vendor environments "
        "(Nokia, Ericsson, Alcatel-Lucent).\n\n"
        "Answer questions using the provided context from telecom documentation. Be specific and technical. "
        "If the context doesn't contain enough information to fully answer, say so clearly. "
        "Always cite which part of the documentation supports your answer."
    )

    user_prompt = (
        f"Context from telecom documentation:\n{context}\n\n"
        f"Question: {question}\n\n"
        "Please provide a detailed, technical answer based on the documentation above."
    )

    client = OpenAI(api_key=key, base_url=DEEPSEEK_BASE_URL)
    resp = client.chat.completions.create(
        model=DEEPSEEK_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,
        max_tokens=1000,
    )

    return resp.choices[0].message.content, sources


# ── Suggestion Buttons ────────────────────────────────────────────────────────
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

question = st.session_state.pop("prefill", None)

user_input = st.chat_input("Ask a telecom engineering question...")
if user_input:
    question = user_input

if question:
    if not api_key:
        st.error("Please enter your DeepSeek API key in the sidebar.")
        st.stop()

    with st.chat_message("user"):
        st.markdown(question)
    st.session_state.messages.append({"role": "user", "content": question})

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
                    "sources": sources,
                })
            except Exception as e:
                st.error(f"Error: {e}")


# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.caption("Telecom RAG Assistant | Supabase pgvector · DeepSeek API · Streamlit")
