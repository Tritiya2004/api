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
    
    return chunks

def create_embeddings_external_api(chunks: List[str]) -> np.ndarray:
    """
    Create embeddings using external API
    """
    headers = {
        "Authorization": f"Bearer {API_TOKEN}",
        "Content-Type": "application/json",
        "Subscription-Key": SUBSCRIPTION_KEY
    }
    
    embeddings = []
    batch_size = 5  # Process in small batches to avoid timeouts
    
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        
        for text in batch:
            payload = {
                "input": text,
                "model": "text-embedding-3-small"
            }
            
            try:
                response = requests.post(
                    EMBEDDING_API_URL,
                    headers=headers,
                    json=payload,
                    timeout=30
                )
                response.raise_for_status()
                
                result = response.json()
                embedding = result["data"][0]["embedding"]
                embeddings.append(embedding)
                
            except Exception as e:
                print(f"Error getting embedding for chunk {i}: {e}")
                # Fallback to TF-IDF for failed chunks
                raise e
    
    return np.array(embeddings).astype("float32")

def create_tfidf_embeddings(chunks: List[str]) -> np.ndarray:
    """
    Create TF-IDF embeddings as fallback
    """
    print("Using TF-IDF embeddings (fallback method)")
    vectorizer = TfidfVectorizer(
        max_features=1000,
        stop_words='english',
        ngram_range=(1, 2)
    )
    
    tfidf_matrix = vectorizer.fit_transform(chunks)
    return tfidf_matrix.toarray().astype("float32")

def search_similar_chunks(
    query: str, 
    chunks: List[str], 
    embeddings: np.ndarray, 
    top_k: int = 5
) -> List[str]:
    """
    Find most similar chunks to the query
    """
    try:
        if SUBSCRIPTION_KEY:
            # Get query embedding from external API
            query_embedding = get_query_embedding_external_api(query)
            
            # Calculate cosine similarity
            similarities = cosine_similarity([query_embedding], embeddings).flatten()
            
        else:
            # Use TF-IDF similarity
            similarities = calculate_tfidf_similarity(query, chunks, embeddings)
            
    except Exception as e:
        print(f"Error in similarity search, using fallback: {e}")
        similarities = calculate_tfidf_similarity(query, chunks, embeddings)
    
    # Get top-k most similar chunks
    top_indices = similarities.argsort()[-top_k:][::-1]
    
    return [chunks[i] for i in top_indices if similarities[i] > 0.1]

def get_query_embedding_external_api(query: str) -> np.ndarray:
    """
    Get embedding for query using external API
    """
    headers = {
        "Authorization": f"Bearer {API_TOKEN}",
        "Content-Type": "application/json",
        "Subscription-Key": SUBSCRIPTION_KEY
    }
    
    payload = {
        "input": query,
        "model": "text-embedding-3-small"
    }
    
    response = requests.post(
        EMBEDDING_API_URL,
        headers=headers,
        json=payload,
        timeout=15
    )
    response.raise_for_status()
    
    result = response.json()
    return np.array(result["data"][0]["embedding"]).astype("float32")

def calculate_tfidf_similarity(
    query: str, 
    chunks: List[str], 
    tfidf_embeddings: np.ndarray
) -> np.ndarray:
    """
    Calculate TF-IDF similarity between query and chunks
    """
    # Create vectorizer with same parameters as chunks
    vectorizer = TfidfVectorizer(
        max_features=1000,
        stop_words='english',
        ngram_range=(1, 2)
    )
    
    # Fit on chunks + query
    all_texts = chunks + [query]
    tfidf_matrix = vectorizer.fit_transform(all_texts)
    
    # Query is the last item
    query_vector = tfidf_matrix[-1:]
    chunk_vectors = tfidf_matrix[:-1]
    
    # Calculate cosine similarity
    similarities = cosine_similarity(query_vector, chunk_vectors).flatten()
    
    return similarities
