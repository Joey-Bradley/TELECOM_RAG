# 📡 Telecom RAG Assistant

A RAG (Retrieval-Augmented Generation) chatbot that answers telecom engineering questions grounded in your own documentation. Built with ChromaDB, sentence-transformers, DeepSeek API, and Streamlit.

## What It Does

Ask natural language questions about telecom engineering and get accurate, source-cited answers pulled directly from your documentation:

- LTE/5G troubleshooting (handover failures, drop calls, poor throughput)
- RF optimization (antenna tuning, interference analysis, coverage planning)
- Network KPI analysis and capacity planning
- CDMA/PCS system concepts
- Multi-vendor environment guidance (Nokia, Ericsson, Alcatel-Lucent)

## Tech Stack

| Component | Technology |
|---|---|
| Vector Store | ChromaDB (local, persistent) |
| Embeddings | sentence-transformers (all-MiniLM-L6-v2, free/local) |
| LLM | DeepSeek API (deepseek-chat) |
| UI | Streamlit |
| Document Parsing | Plain text + PyPDF |

## Setup

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Add your documents
Drop `.txt` or `.pdf` files into the `docs/` folder. A sample telecom knowledge base (`telecom_knowledge.txt`) is included to get you started.

### 3. Index documents
```bash
python ingest.py
```

### 4. Run the app
```bash
streamlit run app.py
```

### 5. Enter your DeepSeek API key
Get a key at [platform.deepseek.com](https://platform.deepseek.com) and enter it in the sidebar.

## Adding More Documents

You can add any telecom documentation:
- 3GPP specification excerpts (copy relevant sections as .txt)
- Vendor runbooks and troubleshooting guides
- Network engineering SOPs
- Your own performance analysis notes

After adding files, re-run `python ingest.py` to re-index.

## Project Structure

```
telecom_rag/
├── app.py              # Streamlit web app
├── ingest.py           # Document ingestion script
├── requirements.txt    # Python dependencies
├── README.md
├── docs/               # Your telecom documents go here
│   └── telecom_knowledge.txt
└── chroma_db/          # Auto-created by ingest.py (vector store)
```

## How RAG Works

1. **Ingest**: Documents are split into chunks, converted to vector embeddings, and stored in ChromaDB
2. **Query**: Your question is embedded and compared against all stored chunks using cosine similarity
3. **Retrieve**: The top 5 most relevant chunks are retrieved
4. **Generate**: DeepSeek uses those chunks as context to generate a grounded, accurate answer
5. **Cite**: Sources are shown so you can verify every answer

## Author

Joey Bradley | [LinkedIn](https://www.linkedin.com/in/joey-bradley-740a2925/) | [GitHub](https://github.com/Joey-Bradley)
