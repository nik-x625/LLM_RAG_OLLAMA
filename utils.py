import re
import yaml
import requests

def extract_metadata_from_mdx(file_path: str):
    with open(file_path, "r") as file:
        content = file.read()
    parts = content.split('---')
    if len(parts) < 3:
        return {}, content
    try:
        metadata = yaml.safe_load(parts[1])
    except yaml.YAMLError:
        return {}, content
    return metadata, '---'.join(parts[2:]).strip()

def clean_article_content(content: str):
    content = re.sub(r"^import .*\n?", "", content, flags=re.MULTILINE)
    content = re.sub(r"<[^>]+>", "", content)
    return content.strip()

def create_chunks(content: str):
    chunks = []
    for chunk in content.split("\n\n"):
        chunks.append(chunk.strip())
    return chunks

def generate_embeddings(text: str):
    response = requests.post(
        "http://host.docker.internal:11434/api/embed",
        json={"model": "mistral", "input": text}
    )
    return response.json().get("embeddings", [None])[0]

def generate_response(prompt: str):
    response = requests.post(
        "http://host.docker.internal:11434/api/generate",
        json={
            "model": "mistral",
            "prompt": prompt,
            "stream": False,
            "options": {"num_ctx": 10000}
        }
    )
    return response.json()["response"]
