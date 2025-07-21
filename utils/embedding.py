import pdfplumber
import numpy as np
from typing import List
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def get_chunks_and_embeddings(pdf_path: str, chunk_size: int = 400):
    """Extract text chunks from PDF and create TF-IDF embeddings"""
    
    # Extract text from PDF
    with pdfplumber.open(pdf_path) as pdf:
        text = " ".join(page.extract_text() or "" for page in pdf.pages)
    
    if not text.strip():
        raise ValueError("No text extracted from PDF")
    
    # Create chunks
    chunks = []
    for i in range(0, len(text), chunk_size):
        chunk = text[i:i+chunk_size].strip()
        if chunk:  # Only add non-empty chunks
            chunks.append(chunk)
    
    if not chunks:
        raise ValueError("No valid chunks created from PDF")
    
    # Create TF-IDF embeddings
    vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
    vectors = vectorizer.fit_transform(chunks)
    
    return chunks, vectors

def search_similar_chunks(query: str, chunks: List[str], vectors, top_k: int = 3):
    """Find most similar chunks using TF-IDF cosine similarity"""
    
    if not chunks or vectors is None:
        return ["No relevant content found"]
    
    # Create TF-IDF vectorizer with same vocabulary
    vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
    
    # Fit on existing chunks to get same vocabulary
    vectorizer.fit(chunks)
    
    # Transform query using same vocabulary
    query_vector = vectorizer.transform([query])
    
    # Calculate cosine similarity
    similarities = cosine_similarity(query_vector, vectors).flatten()
    
    # Get top-k most similar chunks
    if len(similarities) < top_k:
        top_k = len(similarities)
    
    top_indices = np.argsort(similarities)[-top_k:][::-1]
    
    # Return the most relevant chunks
    return [chunks[i] for i in top_indices if similarities[i] > 0.01]  # Filter very low si
