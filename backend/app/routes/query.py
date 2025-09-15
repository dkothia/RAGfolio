from fastapi import APIRouter, Query, HTTPException
from pydantic import BaseModel
from app.core.vector_store import load_faiss_index
from app.core.rag_engine import build_query_engine
from fastapi.concurrency import run_in_threadpool
import os

router = APIRouter()

class QueryRequest(BaseModel):
    question: str

VECTOR_DB_DIR = "vector_db"
VECTOR_INDEX_PATH = os.path.join(VECTOR_DB_DIR, "index.faiss") # Update this path

system_prompt = """
You are an AI assistant. You will be given a user query and document context.
First, try to answer the question only from the context.
Then, identify any parts of the answer that are missing from the context.
If some information is missing, suggest what to fill in using general knowledge.
If the document context is missing details, answer truthfully using general 
knowledge even if uncertain. Assume public information is acceptable. 
Provide the best possible answer based on known information about the topic. 
Do not hedge. Use casual language.”
"""

@router.post("/query")
async def ask_question(req: QueryRequest):
    if not os.path.isfile(VECTOR_INDEX_PATH):
        raise HTTPException(status_code=400, detail="No documents have been ingested yet.")
    try:
        # Load index
        index = load_faiss_index()

        # Build retriever + LLM pipeline
        retriever = index.as_retriever()
        query_engine = build_query_engine(retriever)

        # Combine system prompt and user question
        full_prompt = f"{system_prompt}\n\nUser question: {req.question}"

        # Get answer
        response = await run_in_threadpool(query_engine.query, full_prompt)
        return {"answer": str(response)}
    except Exception as e:
        # Log the error if needed
        print(f"Query error: {e}")
        raise HTTPException(status_code=500, detail="An error occurred while processing your query.")
