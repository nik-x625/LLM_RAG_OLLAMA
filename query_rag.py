import requests
from qdrant_client import QdrantClient
from utils import generate_embeddings, generate_response
import time
from datetime import datetime
from config import QDRANT_URL, COLLECTION_NAME, MODEL_NAME

client = QdrantClient(url=QDRANT_URL)

def get_timestamp():
    return datetime.now().strftime("%H:%M:%S")

def process_query(prompt: str):
    start_time = time.time()
    print(f"\n[{get_timestamp()}] Starting query processing...")

    # Generate embeddings for the prompt
    print(f"[{get_timestamp()}] Generating embeddings...")
    embed_start = time.time()
    adjusted_prompt = f"Represent this sentence for searching relevant passages: {prompt}"
    embeddings = generate_embeddings(adjusted_prompt, MODEL_NAME)
    embed_time = time.time() - embed_start
    print(f"[{get_timestamp()}] Embeddings generated in {embed_time:.2f} seconds")

    # Query Qdrant for relevant passages
    print(f"[{get_timestamp()}] Searching Qdrant...")
    search_start = time.time()
    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=embeddings,
        with_payload=True,
        limit=10
    )
    search_time = time.time() - search_start
    print(f"[{get_timestamp()}] Search completed in {search_time:.2f} seconds")

    # Format the results
    format_start = time.time()
    relevant_passages = "\n".join(
        [f"- Article Title: {point.payload['title']} -- Article Slug: {point.payload['slug']} -- Article Content: {point.payload['content']}" for point in results.points])
    format_time = time.time() - format_start

    # Augment the prompt with retrieved context
    print(f"[{get_timestamp()}] Generating response...")
    response_start = time.time()
    augmented_prompt = f"""
      The following are relevant passages:
      <retrieved-data>
      {relevant_passages}
      </retrieved-data>

      Here's the original user prompt, answer with help of the retrieved passages:
      <user-prompt>
      {prompt}
      </user-prompt>
    """

    # Generate and print response
    response = generate_response(augmented_prompt, MODEL_NAME)
    response_time = time.time() - response_start
    print(f"[{get_timestamp()}] Response generated in {response_time:.2f} seconds")

    # Print final timing summary
    total_time = time.time() - start_time
    print("\nTiming Summary:")
    print(f"- Embedding generation: {embed_time:.2f} seconds")
    print(f"- Qdrant search: {search_time:.2f} seconds")
    print(f"- Response generation: {response_time:.2f} seconds")
    print(f"- Total processing time: {total_time:.2f} seconds")
    print("\nAnswer:", response, "\n")

def main():
    print("Welcome to the RAG Q&A system!")
    print("Type 'exit' or 'quit' to end the session.")
    print("----------------------------------------")
    
    while True:
        try:
            prompt = input("\nEnter your question: ").strip()
            
            if prompt.lower() in ['exit', 'quit']:
                print("\nThank you for using the RAG Q&A system. Goodbye!")
                break
                
            if not prompt:
                print("Please enter a question.")
                continue
                
            process_query(prompt)
            
        except KeyboardInterrupt:
            print("\n\nSession terminated by user. Goodbye!")
            break
        except Exception as e:
            print(f"\nAn error occurred: {str(e)}")
            print("Please try again.")

if __name__ == "__main__":
    main()