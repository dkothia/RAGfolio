from fastapi import APIRouter
from app.core.vector_store import load_faiss_index
from app.core.rag_engine import build_query_engine

router = APIRouter()

@router.get("/summary")
async def summarize_documents():
    # Load FAISS index
    index = load_faiss_index()

    # Build retriever query engine
    query_engine = build_query_engine(index.as_retriever())

    # Run summarization with a fixed prompt
    response = query_engine.query("Summarize the document in 100 words or more")
    return {"summary": str(response)}
