# Local RAG Pipeline Setup Guide

Notes to setting up a local RAG (Retrieval-Augmented Generation) pipeline using Ollama, Qdrant, and LlamaIndex.

## Prerequisites

- Linux/macOS/WSL system with:
  - ≥ 16 GB RAM (32 GB recommended)
  - > 50 GB disk space
- Docker or native runtimes for:
  - Ollama
  - Vector database (Qdrant)
- Python 3.10+ with virtual environment support
- Your documents in any format (PDF, Word, HTML, Markdown, etc.)

## Architecture Overview

The pipeline consists of:
- Ollama (LLM + embeddings)
- Qdrant (vector store)
- LangChain/LlamaIndex (orchestration)
- FastAPI (endpoint)

## Setup Steps

### 1. Set Up Ollama LLM Service

```bash
# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Pull a model (choose one)
ollama pull mistral     # 7B parameters
# or
ollama pull llama2:7b   # 4-bit quantized

# Start Ollama service
ollama serve &          # Available at http://localhost:11434
```

Test the installation:
```bash
curl http://localhost:11434/api/generate \
  -d '{"model":"mistral","prompt":"hello"}'
```

### 2. Set Up Vector Database

Choose one of these options:

| Option    | Command                                    | Notes                          |
|-----------|--------------------------------------------|--------------------------------|
| FAISS     | In-process; no server                      | Easiest, but single-process    |
| Qdrant    | `docker run -p 6333:6333 qdrant/qdrant`    | Good defaults, simple REST     |
| Weaviate  | `docker run -p 8080:8080 semitechnologies/weaviate` | GraphQL API, extra modules |

This guide uses Qdrant on port 6333.

### 3. Set Up Python Environment and Dependencies

```bash
# Create and activate virtual environment
python -m venv rag-env
source rag-env/bin/activate

# Install required packages
pip install llama-index qdrant-client python-docx pypdf pillow beautifulsoup4 tqdm
```

### 4. Create Document Ingestion Script

Create `ingest.py`:
```python
from llama_index import (
    download_loader, VectorStoreIndex, ServiceContext,
    StorageContext, set_global_service_context
)
from llama_index.embeddings.ollama import OllamaEmbedding
from qdrant_client import QdrantClient

# Set up document loaders
PDFReader   = download_loader("PDFReader")
DocxReader  = download_loader("DocxReader")
SimpleHTML  = download_loader("BeautifulSoupWebReader")

# Load documents
docs = []
for path in Path("docs").rglob("*"):
    if path.suffix.lower() == ".pdf":
        docs.extend(PDFReader().load_data(str(path)))
    elif path.suffix.lower() in [".docx", ".doc"]:
        docs.extend(DocxReader().load_data(str(path)))
    elif path.suffix.lower() in [".html", ".htm"]:
        docs.extend(SimpleHTML().load_data(file=path))

# Configure embeddings
embeddings = OllamaEmbedding(
    model_name="mistral", base_url="http://localhost:11434"
)

service_context = ServiceContext.from_defaults(embed_model=embeddings)
set_global_service_context(service_context)

# Set up Qdrant index
client = QdrantClient(host="localhost", port=6333)
storage = StorageContext.from_defaults(vector_store=client, namespace="prod-docs")

index = VectorStoreIndex.from_documents(
    docs,
    storage_context=storage,
    show_progress=True,
)
index.storage_context.persist()
```

Run the ingestion:
```bash
python ingest.py
```

### 5. Create FastAPI Endpoint

Install FastAPI:
```bash
pip install fastapi uvicorn
```

Create `api.py`:
```python
from fastapi import FastAPI
from pydantic import BaseModel
from llama_index import VectorStoreIndex, StorageContext, ServiceContext
from llama_index.embeddings.ollama import OllamaEmbedding
from qdrant_client import QdrantClient

app = FastAPI(title="Product-Docs Q&A")

class Query(BaseModel):
    question: str

# Initialize index on startup
client = QdrantClient(host="localhost", port=6333)
storage = StorageContext.from_defaults(vector_store=client, namespace="prod-docs")
embeddings = OllamaEmbedding(model_name="mistral", base_url="http://localhost:11434")
service_context = ServiceContext.from_defaults(embed_model=embeddings)
index = VectorStoreIndex.from_vector_store(
    storage_context=storage,
    service_context=service_context,
)
qa_engine = index.as_query_engine(streaming=False)

@app.post("/ask")
def ask(q: Query):
    resp = qa_engine.query(q.question)
    return {
        "answer": str(resp),
        "sources": [
            {"text": s.node.text[:200], "id": s.node.node_id}
            for s in resp.source_nodes
        ]
    }
```

Start the API:
```bash
uvicorn api:app --host 0.0.0.0 --port 8000
```

### 6. Test the Pipeline

```bash
curl -X POST http://localhost:8000/ask \
     -H "Content-Type: application/json" \
     -d '{"question":"What monitoring options does Product X have since 2025?"}'
```

## Production Considerations

| Area           | Quick Win                                | When to Implement              |
|----------------|------------------------------------------|--------------------------------|
| Chunking       | Tune chunk size/overlap in loaders       | After initial testing          |
| Caching        | Store embeddings in local SQLite         | For large document corpora     |
| Authentication | Add JWT/OAuth to FastAPI routes          | Before public exposure         |
| Scaling        | Use vLLM/TGI; multiple Qdrant shards     | When QPS > 5                   |
| Metadata       | Add product/version/year tags            | For complex repositories       |
| UI             | Build chat frontend (React/Streamlit)    | For demos                      |

## Benefits

- Fully local RAG pipeline
- Modular architecture allows easy component swapping
- No external API dependencies
- Scalable and customizable