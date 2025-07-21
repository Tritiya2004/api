import os
import tempfile
import requests
import pdfplumber
import time
from fastapi import FastAPI, Request, Header, HTTPException
from pydantic import BaseModel
from utils.embedding import get_chunks_and_embeddings, search_similar_chunks
from utils.llm import generate_answer
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="LLM Query Retrieval API", version="1.0.0")

# API Key for authentication
API_KEY = os.getenv("API_KEY", "sk-spgw-api01-b39ff15132692a6834835a552e6f65b3")

class QueryRequest(BaseModel):
    documents: str
    questions: list[str]

class QueryResponse(BaseModel):
    answers: list[str]

@app.post("/hackrx/run", response_model=QueryResponse)
async def hackrx_run(
    req: Request, 
    body: QueryRequest, 
    authorization: str = Header(None)
):
    """
    Process document and answer questions using LLM-powered retrieval
    """
    start_time = time.time()
    
    # Authentication check
    if authorization is None or authorization != f"Bearer {API_KEY}":
        raise HTTPException(status_code=401, detail="Unauthorized.")
    
    try:
        # Download and parse PDF
        print(f"Downloading PDF from: {body.documents}")
        response = requests.get(body.documents, timeout=30)
        response.raise_for_status()
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(response.content)
            tmp_path = tmp.name

        # Extract chunks and create embeddings
        print("Processing PDF and creating embeddings...")
        chunks, vectors = get_chunks_and_embeddings(tmp_path)
        
        answers = []
        print(f"Processing {len(body.questions)} questions...")
        
        for i, question in enumerate(body.questions):
            print(f"Processing question {i+1}: {question[:50]}...")
            
            # Find relevant chunks
            relevant_chunks = search_similar_chunks(question, chunks, vectors)
      
