import os
import uuid
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from utils import extract_metadata_from_mdx, clean_article_content, create_chunks, generate_embeddings

client = QdrantClient(url="http://qdrant:6333")

if not client.collection_exists(collection_name="articles"):
    client.create_collection(
        collection_name="articles",
        vectors_config=VectorParams(size=4096, distance=Distance.COSINE),
    )

def store_article(metadata: dict, chunks: list[str]):
    for chunk in chunks:
        chunk_id = str(uuid.uuid4())
        adjusted_metadata = {**metadata, "content": chunk}
        embeddings = generate_embeddings(chunk)
        if embeddings:
            client.upsert(
                collection_name="articles",
                wait=True,
                points=[PointStruct(id=chunk_id, vector=embeddings, payload=adjusted_metadata)],
            )

def main():
    article_files = [f for f in os.listdir("docs") if f.endswith(".html")]
    for article_file in article_files:
        file_path = os.path.join("docs", article_file)
        metadata, article_content = extract_metadata_from_mdx(file_path)
        cleaned = clean_article_content(article_content)
        chunks = create_chunks(cleaned)
        metadata["slug"] = article_file.replace(".html", "")
        store_article(metadata, chunks)

if __name__ == "__main__":
    main()