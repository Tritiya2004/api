import requests
import pdfplumber
import numpy as np
import os
import hashlib
from typing import List, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# External API configuration
EMBEDDING_API_URL = "https://dev-api.healthrx.co.in/sp-gw/api/openai/v1/embeddings"
API_TOKEN = os.getenv("API_TOKEN", "sk-spgw-api01-key")
SUBSCRIPTION_KEY = os.getenv("SUBSCRIPTION_KEY", "")

def get_chunks_and_embeddings(pdf_path: str, chunk_size: int = 500) -> Tuple[List[str], np.ndarray]:
    """
    Extract text from PDF, chunk it, and create embeddings
    """
    print("Extracting text from PDF...")
    
    # Extract text from PDF
    with pdfplumber.open(pdf_path) as pdf:
        full_text = ""
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                full_text += text + " "
    
    print(f"Extracted {len(full_text)} characters from PDF")
    
    # Chunk the text
    chunks = create_text_chunks(full_text, chunk_size)
    print(f"Created {len(chunks)} chunks")
    
    # Create embeddings using TF-IDF (fallback) or external API
    try:
        if SUBSCRIPTION_KEY:
            embeddings = create_embeddings_external_api(chunks)
        else:
            embeddings = create_tfidf_embeddings(chunks)
    except Exception as e:
        print(f"Failed to create embeddings via API, falling back to TF-IDF: {e}")
        embeddings = create_tfidf_embeddings(chunks)
    
    return chunks, embeddings

def create_text_chunks(text: str, chunk_size: int = 500) -> List[str]:
    """
    Split text into overlapping chunks
    """
    words = text.split()
    chunks = []
    overlap = chunk_size // 4  # 25% overlap
    
    for i in range(0, len(words), chunk_size - overlap):
        chunk_words = words[i:i + chunk_size]
        chunk = " ".join(chunk_words)
        if len(chunk.strip()) > 50:  # Only include meaningful chunks
            chunks.append(chunk.strip())
