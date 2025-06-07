# # ingest.py
# import os
# from llama_index import VectorStoreIndex, SimpleDirectoryReader, ServiceContext
# from llama_index.vector_stores.qdrant import QdrantVectorStore
# from llama_index.embeddings.openai import OpenAIEmbedding
# from qdrant_client import QdrantClient

# qdrant_host = os.getenv("QDRANT_HOST", "localhost")
# qdrant_port = int(os.getenv("QDRANT_PORT", 6333))

# client = QdrantClient(host=qdrant_host, port=qdrant_port)

# documents = SimpleDirectoryReader("./docs").load_data()
# vector_store = QdrantVectorStore(client=client, collection_name="my_docs")
# service_context = ServiceContext.from_defaults(embed_model=OpenAIEmbedding())
# index = VectorStoreIndex.from_documents(
#     documents, service_context=service_context, vector_store=vector_store
# )
# index.storage_context.persist()




# from llama_index import (
#     download_loader, VectorStoreIndex, ServiceContext,
#     StorageContext, set_global_service_context
# )
from llama_index.embeddings.ollama import OllamaEmbedding

from llama_index.core import set_global_service_context

#from llama_index.core import ServiceContext

#from llama_index.core import settings as global_settings


#from llama_index.core.settings import Settings

#from llama_index.core import settings
#from llama_index.core import settings# as global_settings
#from llama_index.core import VectorStoreIndex, StorageContext






from qdrant_client import QdrantClient

# 3‑A. choose loaders for each format
# PDFReader   = download_loader("PDFReader")
# DocxReader  = download_loader("DocxReader")
# SimpleHTML  = download_loader("BeautifulSoupWebReader")




from bs4 import BeautifulSoup
from llama_index.core.schema import Document

def read_html_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        html = f.read()
    soup = BeautifulSoup(html, 'html.parser')
    text = soup.get_text(separator='\n')
    return Document(text=text, extra_info={'source': filepath})

# Example usage
doc = read_html_file('docs/doc1-cave.html')
print(doc.text)






docs = []
# for path in Path("docs").rglob("*"):
#     #if path.suffix.lower() == ".pdf":
#     #    docs.extend(PDFReader().load_data(str(path)))
#     #elif path.suffix.lower() in [".docx", ".doc"]:
#     #    docs.extend(DocxReader().load_data(str(path)))
#     #elif path.suffix.lower() in [".html", ".htm"]:
#     docs.extend(SimpleHTML().load_data(file=path))

docs.append(doc)

# 3‑B. connect embedding model to Ollama
embeddings = OllamaEmbedding(
    model_name="mistral", base_url="http://localhost:11434"
)



#from llama_index.core.settings import Settings
#from llama_index.core.settings import Settings as global_settings

from llama_index.core import settings
from llama_index.core import StorageContext
from llama_index.core import VectorStoreIndex


#my_settings = Settings(embed_model=embeddings)
#global_settings.embed_model = my_settings.embed_model




#settings.embed_model = embeddings 

# 3‑C. set up Qdrant-backed index
client = QdrantClient(host="localhost", port=6333)
storage = StorageContext.from_defaults(vector_store=client)

# index = VectorStoreIndex.from_documents(
#     docs,
#     storage_context=storage,
#     show_progress=True,
# )


from llama_index.core import ServiceContext

service_context = ServiceContext.from_defaults(embed_model=embeddings)

index = VectorStoreIndex.from_documents(
    docs,
    storage_context=storage,
    service_context=service_context,
    show_progress=True,
)


index.storage_context.persist()
