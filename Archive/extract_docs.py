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