import os
import tempfile
import requests
import pdfplumber
from fastapi import FastAPI, Request, Header, HTTPException
from pydantic import BaseModel
from utils.embedding import get_chunks_and_embeddings, search_similar_chunks
from utils.llm import generate_answer
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()
API_KEY = "your_internal_api_key"   # Replace with your secret

class QueryRequest(BaseModel):
    documents: str
    questions: list[str]

@app.post("/hackrx/run")
async def hackrx_run(req: Request, body: QueryRequest, authorization: str = Header(None)):
    # Check Bearer token
    if authorization is None or authorization != f"Bearer {API_KEY}":
        raise HTTPException(status_code=401, detail="Unauthorized.")

    # Download and parse PDF
    pdf_url = body.documents
    r = requests.get(pdf_url)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(r.content)
        tmp_path = tmp.name

    chunks, vectors = get_chunks_and_embeddings(tmp_path)

    answers = []
    for q in body.questions:
        best_chunks = search_similar_chunks(q, chunks, vectors)
        answer = generate_answer(q, best_chunks)
        answers.append(answer)

    return {"answers": answers}
