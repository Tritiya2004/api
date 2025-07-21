import numpy as np
from typing import List

def get_chunks_and_embeddings(pdf_path: str, chunk_size: int = 400):
    import pdfplumber
    with pdfplumber.open(pdf_path) as pdf:
        text = " ".join(page.extract_text() or "" for page in pdf.pages)
    chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
    
    # Use simple hash-based vector for embedding
    embeddings = [simple_text_embedding(chunk) for chunk in chunks]
    return chunks, np.array(embeddings).astype("float32")

def simple_text_embedding(text: str, size: int = 300):
    # Hash each word into embedding
    words = text.lower().split()
    vec = np.zeros(size)
    for word in words:
        idx = hash(word) % size
        vec[idx] +=1
    # Normalize
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec /= norm
    return vec
