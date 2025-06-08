import os
import time
import uuid
from datetime import datetime
import fitz  # PyMuPDF
from pdf2image import convert_from_path
import pytesseract
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer
from utils import generate_embeddings

from config import PDF_COLLECTION_NAME, QDRANT_URL, VECTOR_SIZE, MODEL_NAME, PDF_DIRECTORY

# Load embedding model
embedder = SentenceTransformer('all-MiniLM-L6-v2')  # Optional: replace with your model

# Init Qdrant client
client = QdrantClient(url=QDRANT_URL)

def get_timestamp():
    return datetime.now().strftime("%H:%M:%S")

def extract_text_from_pdf(path):
    doc = fitz.open(path)
    text = ""
    for page in doc:
        page_text = page.get_text()
        text += page_text if page_text.strip() else ""
    return text.strip()

def fallback_ocr(path):
    images = convert_from_path(path)
    text = ""
    for img in images:
        text += pytesseract.image_to_string(img)
    return text

def smart_extract_text(path):
    text = extract_text_from_pdf(path)
    if not text or len(text) < 100:
        print(f"[{get_timestamp()}] Fallback to OCR for {os.path.basename(path)}")
        text = fallback_ocr(path)
    return text

def create_chunks(text, chunk_size=500, chunk_overlap=50):
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    return splitter.split_text(text)

def initialize_collection():
    print(f"[{get_timestamp()}] Initializing PDF collection...")
    if not client.collection_exists(collection_name=PDF_COLLECTION_NAME):
        client.create_collection(
            collection_name=PDF_COLLECTION_NAME,
            vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE)
        )
    print(f"[{get_timestamp()}] Collection ready: {PDF_COLLECTION_NAME}")

def store_chunks(chunks, metadata):
    for i, chunk in enumerate(chunks):
        chunk_id = str(uuid.uuid4())
        embedding = generate_embeddings(chunk, MODEL_NAME)
        client.upsert(
            collection_name=PDF_COLLECTION_NAME,
            points=[
                PointStruct(
                    id=chunk_id,
                    vector=embedding,
                    payload={
                        **metadata,
                        "chunk_index": i,
                        "content": chunk
                    }
                )
            ],
            wait=True
        )

def ingest_pdfs():
    print(f"[{get_timestamp()}] Starting PDF ingestion...")
    initialize_collection()

    files = [f for f in os.listdir(PDF_DIRECTORY) if f.lower().endswith(".pdf")]
    print(f"[{get_timestamp()}] Found {len(files)} PDF files.")

    for i, pdf_file in enumerate(files, 1):
        print(f"\n[{get_timestamp()}] Processing file {i}/{len(files)}: {pdf_file}")
        path = os.path.join(PDF_DIRECTORY, pdf_file)
        
        start = time.time()
        raw_text = smart_extract_text(path)
        chunks = create_chunks(raw_text)
        
        metadata = {
            "filename": pdf_file,
            "slug": pdf_file.replace(".pdf", ""),
            "source": "pdf"
        }

        store_chunks(chunks, metadata)
        print(f"[{get_timestamp()}] Completed {pdf_file} in {time.time() - start:.2f} seconds")

    print(f"\n[{get_timestamp()}] PDF ingestion completed.")

if __name__ == "__main__":
    ingest_pdfs()
