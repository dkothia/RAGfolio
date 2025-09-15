from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from app.core.vector_store import load_faiss_index
from app.core.rag_engine import build_query_engine
import os

router = APIRouter()

class ChartRequest(BaseModel):
    prompt: str  # User's chart request

@router.post("/charts")
async def extract_chart_data(req: ChartRequest):
    # Load index
    index = load_faiss_index()

    # Build retriever + LLM pipeline using rag_engine
    retriever = index.as_retriever(similarity_top_k=8)
    query_engine = build_query_engine(retriever)

    # Retrieve relevant nodes
    nodes = retriever.retrieve(req.prompt)
    if not nodes:
        raise HTTPException(status_code=404, detail="No relevant chartable data found for your request.")

    # Combine content for context
    content = "\n\n".join([n.get_content() for n in nodes])
    chart_prompt = (
        f"{req.prompt}\n\n"
        "Extract key numerical or tabular insights from the above context that could be plotted. "
        "Return them as structured JSON (e.g., list of dicts with x/y fields)."
        "\n\n"
        f"{content}"
    )

    # Ask LLM to return chartable JSON using the query engine
    response = query_engine.query(chart_prompt)

    return {"chart_data": str(response).strip()}
