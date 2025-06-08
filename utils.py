import re
import yaml
import requests
from config import OLLAMA_URL, MODEL_NAME, CONTEXT_WINDOW


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

def generate_embeddings(text: str, model_name: str = MODEL_NAME):
    response = requests.post(
        f"{OLLAMA_URL}/api/embed",
        json={"model": model_name, "input": text}
    )
    return response.json().get("embeddings", [None])[0]

def generate_response(prompt: str, model_name: str = MODEL_NAME):
    response = requests.post(
        f"{OLLAMA_URL}/api/generate",
        json={
            "model": model_name,
            "prompt": prompt,
            "stream": False,
            "options": {"num_ctx": CONTEXT_WINDOW}
        }
    )
    return response.json()["response"]
