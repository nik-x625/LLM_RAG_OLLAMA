# api.py
from fastapi import FastAPI
from pydantic import BaseModel
from llama_index import VectorStoreIndex, StorageContext, ServiceContext
from llama_index.embeddings.ollama import OllamaEmbedding
from qdrant_client import QdrantClient

app = FastAPI(title="Product‑Docs Q&A")

class Query(BaseModel):
    question: str

# reload index once on startup
import os
client = QdrantClient(
    host=os.getenv("QDRANT_HOST", "localhost"),
    port=int(os.getenv("QDRANT_PORT", "6333"))
)

storage = StorageContext.from_defaults(vector_store=client, namespace="prod-docs")
embeddings = OllamaEmbedding(model_name="mistral", base_url="http://localhost:11434")
service_context = ServiceContext.from_defaults(embed_model=embeddings)
index = VectorStoreIndex.from_vector_store(
    storage_context=storage,
    service_context=service_context,
)
qa_engine = index.as_query_engine(streaming=False)  # RAG!

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
