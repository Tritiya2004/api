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
API_KEY = os.getenv("API_KEY", "sk-spgw-api01-b39ff15132692a6834835a552e6f65b3")

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
    try:
        r = requests.get(pdf_url, timeout=30)
        r.raise_for_status()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to download PDF: {str(e)}")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(r.content)
        tmp_path = tmp.name

    try:
        chunks, vectors = get_chunks_and_embeddings(tmp_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process PDF: {str(e)}")
    finally:
        # Clean up temp file
        os.unlink(tmp_path)

    answers = []
    for question in body.questions:
        try:
            best_chunks = search_similar_chunks(question, chunks, vectors)
            answer = generate_answer(question, best_chunks)
            answers.append(answer)
        except Exception as e:
            answers.append(f"Error processing question: {str(e)}")

    return {"answers": answers}

@app.get("/")
async def root():
    return {"message": "LLM Query API is running"}
