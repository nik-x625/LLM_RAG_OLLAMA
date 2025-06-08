import requests
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
import yaml
import re
import os
import uuid
import time
from datetime import datetime
from utils import generate_embeddings, extract_metadata_from_mdx, clean_article_content, create_chunks
from config import QDRANT_URL, COLLECTION_NAME, VECTOR_SIZE, MODEL_NAME

client = QdrantClient(url=QDRANT_URL)

def get_timestamp():
    return datetime.now().strftime("%H:%M:%S")

def initialize_collection():
    print(f"[{get_timestamp()}] Initializing Qdrant collection...")
    start_time = time.time()
    
    if not client.collection_exists(collection_name=COLLECTION_NAME):
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
        )
    
    init_time = time.time() - start_time
    print(f"[{get_timestamp()}] Collection initialized in {init_time:.2f} seconds")

def store_article(metadata: dict, chunks: list[str]):
    print(f"[{get_timestamp()}] Storing article: {metadata.get('title', 'Untitled')}")
    start_time = time.time()
    total_chunks = len(chunks)
    
    for i, chunk in enumerate(chunks, 1):
        chunk_start = time.time()
        # Generate a unique ID for each chunk
        chunk_id = str(uuid.uuid4())
        adjusted_metadata = {
            **metadata,
            "content": chunk
        }
        
        # Generate embeddings
        print(f"[{get_timestamp()}] Generating embeddings for chunk {i}/{total_chunks}...")
        embed_start = time.time()
        embeddings = generate_embeddings(chunk, MODEL_NAME)
        embed_time = time.time() - embed_start
        
        if embeddings is not None:
            # Store in Qdrant
            print(f"[{get_timestamp()}] Storing chunk {i}/{total_chunks} in Qdrant...")
            store_start = time.time()
            client.upsert(
                collection_name=COLLECTION_NAME,
                wait=True,
                points=[PointStruct(
                    id=chunk_id, vector=embeddings,
                    payload=adjusted_metadata
                )],
            )
            store_time = time.time() - store_start
            
            chunk_time = time.time() - chunk_start
            print(f"[{get_timestamp()}] Chunk {i}/{total_chunks} processed in {chunk_time:.2f} seconds")
            print(f"  - Embedding generation: {embed_time:.2f} seconds")
            print(f"  - Qdrant storage: {store_time:.2f} seconds")
    
    total_time = time.time() - start_time
    print(f"[{get_timestamp()}] Article processing completed in {total_time:.2f} seconds")

def ingest_documents():
    start_time = time.time()
    print(f"[{get_timestamp()}] Starting document ingestion...")
    
    initialize_collection()
    
    # Store articles
    article_files = [f for f in os.listdir("docs") if f.endswith(".html")]
    total_files = len(article_files)
    print(f"[{get_timestamp()}] Found {total_files} files to process")
    
    for i, article_file in enumerate(article_files, 1):
        file_start = time.time()
        print(f"\n[{get_timestamp()}] Processing file {i}/{total_files}: {article_file}")
        
        file_path = os.path.join("docs", article_file)
        
        # Extract metadata and content
        print(f"[{get_timestamp()}] Extracting metadata and content...")
        extract_start = time.time()
        metadata, article_content = extract_metadata_from_mdx(file_path)
        extract_time = time.time() - extract_start
        
        # Clean content
        print(f"[{get_timestamp()}] Cleaning content...")
        clean_start = time.time()
        cleaned_article_content = clean_article_content(article_content)
        clean_time = time.time() - clean_start
        
        # Create chunks
        print(f"[{get_timestamp()}] Creating chunks...")
        chunk_start = time.time()
        chunks = create_chunks(cleaned_article_content)
        chunk_time = time.time() - chunk_start
        
        metadata["slug"] = article_file.replace(".html", "")
        
        # Store article
        store_article(metadata=metadata, chunks=chunks)
        
        file_time = time.time() - file_start
        print(f"\n[{get_timestamp()}] File {i}/{total_files} completed in {file_time:.2f} seconds")
        print(f"  - Metadata extraction: {extract_time:.2f} seconds")
        print(f"  - Content cleaning: {clean_time:.2f} seconds")
        print(f"  - Chunk creation: {chunk_time:.2f} seconds")
    
    total_time = time.time() - start_time
    print(f"\n[{get_timestamp()}] Document ingestion completed in {total_time:.2f} seconds")

if __name__ == "__main__":
    ingest_documents()