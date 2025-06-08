"""
Configuration settings for the RAG system.
"""

# Model configurations
MODELS = {
    "llama3.2": {
        "name": "llama3.2",
        "vector_size": 3072,
    },
    "mistral": {
        "name": "mistral",
        "vector_size": 4096,
    }
}

# Current model selection
CURRENT_MODEL = "llama3.2"

# Get current model settings
MODEL_NAME = MODELS[CURRENT_MODEL]["name"]
VECTOR_SIZE = MODELS[CURRENT_MODEL]["vector_size"]

# Qdrant settings
QDRANT_URL = "http://qdrant:6333"
COLLECTION_NAME = "articles"

# Ollama settings
OLLAMA_URL = "http://host.docker.internal:11434"
CONTEXT_WINDOW = 10000 

# PDF import related configs
PDF_COLLECTION_NAME = "pdf_documents"
PDF_DIRECTORY = "docs"