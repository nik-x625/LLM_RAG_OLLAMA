# api.py
import os
from fastapi import FastAPI, Query
from llama_index import VectorStoreIndex, StorageContext
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

app = FastAPI()

qdrant_host = os.getenv("QDRANT_HOST", "localhost")
qdrant_port = int(os.getenv("QDRANT_PORT", 6333))

client = QdrantClient(host=qdrant_host, port=qdrant_port)
vector_store = QdrantVectorStore(client=client, collection_name="my_docs")
storage_context = StorageContext.from_defaults(vector_store=vector_store)
index = VectorStoreIndex.from_vector_store(vector_store, storage_context=storage_context)
query_engine = index.as_query_engine()

@app.get("/query")
def query_docs(q: str = Query(..., description="Your question")):
    response = query_engine.query(q)
    return {"response": str(response)}
