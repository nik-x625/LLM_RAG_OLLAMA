# ingest.py
from llama_index import (
    download_loader, VectorStoreIndex, ServiceContext,
    StorageContext, set_global_service_context
)
from llama_index.embeddings.ollama import OllamaEmbedding
from qdrant_client import QdrantClient

# 3‑A. choose loaders for each format
PDFReader   = download_loader("PDFReader")
DocxReader  = download_loader("DocxReader")
SimpleHTML  = download_loader("BeautifulSoupWebReader")

docs = []
for path in Path("docs").rglob("*"):
    if path.suffix.lower() == ".pdf":
        docs.extend(PDFReader().load_data(str(path)))
    elif path.suffix.lower() in [".docx", ".doc"]:
        docs.extend(DocxReader().load_data(str(path)))
    elif path.suffix.lower() in [".html", ".htm"]:
        docs.extend(SimpleHTML().load_data(file=path))

# 3‑B. connect embedding model to Ollama
embeddings = OllamaEmbedding(
    model_name="mistral", base_url="http://localhost:11434"
)

service_context = ServiceContext.from_defaults(embed_model=embeddings)
set_global_service_context(service_context)

# 3‑C. set up Qdrant-backed index

import os
client = QdrantClient(
    host=os.getenv("QDRANT_HOST", "localhost"),
    port=int(os.getenv("QDRANT_PORT", "6333"))
)

storage = StorageContext.from_defaults(vector_store=client, namespace="prod-docs")

index = VectorStoreIndex.from_documents(
    docs,
    storage_context=storage,
    show_progress=True,
)
index.storage_context.persist()
