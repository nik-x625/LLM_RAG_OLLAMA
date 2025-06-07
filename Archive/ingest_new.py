from llama_index.core import (
    download_loader, VectorStoreIndex, ServiceContext,
    StorageContext, set_global_service_context
)
from llama_index.embeddings.ollama import OllamaEmbedding
from qdrant_client import QdrantClient 