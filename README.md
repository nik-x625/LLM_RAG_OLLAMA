# 🧠 RAG System with Qdrant + Ollama (Mistral)

This project implements a modular Retrieval-Augmented Generation (RAG) pipeline.

---

## 🔧 Components

### `index_articles.py` — Indexing
- Cleans & chunks articles
- Generates embeddings
- Stores in Qdrant DB

### `query_rag.py` — Retrieval
- Embeds user query
- Searches Qdrant for relevant chunks
- Calls LLM with retrieved context

## 🧪 Run

### 1. Index articles

```bash
python index_articles.py
```

### 2. Ask a question

```bash
python query_rag.py
```

---

## 📦 Structure

```
rag_project/
├── docs/
├── index_articles.py
├── query_rag.py
├── utils.py
└── README.md
```
## 📊 Workflow Diagram

```mermaid
flowchart TD
    A[HTML/MDX Files] -->|Extract Metadata| B[Clean + Chunk]
    B -->|Process| C[Generate Embeddings]
    C -->|Store| D[Qdrant Database]
    E[User Prompt] -->|Embed| F[Search Query]
    F -->|Find| G[Retrieve Top K]
    G -->|Context| H[Generate Answer]
    H -->|Return| I[Final Response]
```