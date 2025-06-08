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

---

## 📊 Workflow Diagram (Mermaid)

```mermaid
graph TD
    A[HTML/MDX Files] -->|Extract Metadata| B[Clean + Chunk]
    B --> C[Generate Embeddings (Ollama)]
    C --> D[Store in Qdrant]

    E[User Prompt] --> F[Embed Prompt (Ollama)]
    F --> G[Search Qdrant]
    G --> H[Retrieve Top K Chunks]
    H --> I[Send to Mistral with Prompt]
    I --> J[Answer Generated]
```

---

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