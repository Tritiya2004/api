import openai
import pdfplumber
import faiss
import numpy as np
import os

openai.api_key = os.getenv("OPENAI_API_KEY")

def get_chunks_and_embeddings(pdf_path, chunk_size=400):
    with pdfplumber.open(pdf_path) as pdf:
        text = " ".join(page.extract_text() or "" for page in pdf.pages)
    # Chunk the text
    chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
    # Get embeddings
    embeddings = [openai.Embedding.create(input=chunk, model="text-embedding-3-small")["data"][0]["embedding"] for chunk in chunks]
    embeddings = np.array(embeddings).astype("float32")
    return chunks, embeddings

def search_similar_chunks(query, chunks, embeddings, top_k=3):
    # Embed the query
    q_emb = openai.Embedding.create(input=query, model="text-embedding-3-small")["data"][0]["embedding"]
    q_emb = np.array(q_emb).astype("float32").reshape(1,-1)
    # Build FAISS index on the fly
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    D, I = index.search(q_emb, top_k)
    return [chunks[i] for i in I[0]]
