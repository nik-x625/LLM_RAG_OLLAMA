import requests
from qdrant_client import QdrantClient
from utils import generate_embeddings, generate_response

client = QdrantClient(url="http://qdrant:6333")

def main():
    prompt = input("Enter a prompt: ")
    adjusted_prompt = f"Represent this sentence for searching relevant passages: {prompt}"

    embeddings = generate_embeddings(adjusted_prompt)
    results = client.query_points(
        collection_name="articles",
        query=embeddings,
        with_payload=True,
        limit=10
    )

    relevant_passages = "\n".join([
        f"- Title: {p.payload['title']} | Slug: {p.payload['slug']} | Content: {p.payload['content']}"
        for p in results.points
    ])

    augmented_prompt = f"""
    The following are relevant passages:
    <retrieved-data>
    {relevant_passages}
    </retrieved-data>

    Here's the original user prompt:
    <user-prompt>
    {prompt}
    </user-prompt>
    """

    response = generate_response(augmented_prompt)
    print(response)

if __name__ == "__main__":
    main()