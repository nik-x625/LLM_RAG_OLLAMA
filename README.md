# RAG System with Ollama and Qdrant

This project implements a Retrieval-Augmented Generation (RAG) system using Ollama for embeddings and text generation, and Qdrant as the vector database. The system processes HTML documents, creates embeddings, and provides semantic search capabilities.

## Features

- Document processing from HTML files
- Text chunking and cleaning
- Vector embeddings generation using Ollama's Mistral model
- Semantic search using Qdrant vector database
- Question answering with context-aware responses

## Prerequisites

- Docker and Docker Compose
- Ollama running on your host machine
- Python 3.x

## Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd <repository-name>
```

2. Create a `docs` directory and place your HTML documents there:
```bash
mkdir docs
# Add your HTML files to the docs directory
```

3. Start the services using Docker Compose:
```bash
docker-compose up -d
```

## Usage

1. Run the main script:
```bash
python main.py
```

2. The script will:
   - Process all HTML files in the `docs` directory
   - Create embeddings and store them in Qdrant
   - Prompt you to enter a question
   - Return relevant passages and generate an answer

## Architecture

- **Document Processing**: HTML files are processed and split into chunks
- **Embeddings**: Uses Ollama's Mistral model to generate embeddings
- **Vector Storage**: Qdrant stores the embeddings and metadata
- **Search**: Semantic search is performed using cosine similarity
- **Response Generation**: Uses Mistral model to generate context-aware responses

## Configuration

The system uses the following default settings:
- Vector dimension: 4096
- Distance metric: Cosine similarity
- Model: Mistral for both embeddings and text generation

## Directory Structure

```
.
├── docs/               # HTML documents to be processed
├── main.py            # Main application code
├── docker-compose.yml # Docker configuration
└── .gitignore         # Git ignore file
```

## Notes

- The system connects to Ollama running on the host machine using `host.docker.internal`
- Qdrant data is stored in the `qdrant_data` directory (ignored by git)
- Make sure Ollama is running on your host machine before starting the application

## Troubleshooting

If you encounter any issues:
1. Ensure Ollama is running on your host machine
2. Check if the Docker containers are running properly
3. Verify that your HTML documents are properly formatted
4. Check the vector dimensions match between Ollama and Qdrant 