import requests
import pdfplumber
import faiss
import numpy as np
import os

# Remove: import openai

def get_chunks_and_embeddings(pdf_path, chunk_size=400):
    with pdfplumber.open(pdf_path) as pdf:
        text = " ".join(page.extract_text() or "" for page in pdf.pages)
    
    # Chunk the text
    chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
    
    # Use external API for embeddings instead of openai.Embedding.create
    embeddings = []
    for chunk in chunks:
        embedding = get_embedding_from_external_api(chunk)
        embeddings.append(embedding)
    
    embeddings = np.array(embeddings).astype("float32")
    return chunks, embeddings

def get_embedding_from_external_api(text):
    """Get embeddings using external API instead of OpenAI directly"""
    # You may need to implement this based on available embedding endpoint
    # For now, return a dummy embedding or implement with available service
    pass

def search_similar_chunks(query, chunks, embeddings, top_k=3):
    # Get query embedding using external API
    q_emb = get_embedding_from_external_api(query)
    q_emb = np.array(q_emb).astype("float32").reshape(1,-1)
    
    # Build FAISS index
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    D, I = index.search(q_emb, top_k)
    return [chunks[i] for i in I[0]]
