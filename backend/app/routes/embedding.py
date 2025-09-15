from fastapi import APIRouter
from app.core.vector_store import load_faiss_index
import numpy as np
import pandas as pd

router = APIRouter()

@router.get("/embedding")
async def show_embeddings():
    index = load_faiss_index()
    # Get all nodes/chunks
    nodes = index.docstore.docs.values()
    # Get FAISS vectors
    faiss_vectors = index.vector_store._faiss_index.reconstruct_n(0, index.vector_store._faiss_index.ntotal)
    # Prepare data for table
    data = []
    for i, node in enumerate(nodes):
        chunk_text = node.get_content()[:100]  # Show first 100 chars
        vector = faiss_vectors[i].tolist() if i < len(faiss_vectors) else []
        data.append({
            "chunk_id": i,
            "chunk_text": chunk_text,
            "vector": vector
        })
    # Convert to DataFrame for pretty printing (optional)
    df = pd.DataFrame(data)
    # Print table to console (for debugging)
    print(df)
    # Return as JSON for API
    return df.to_dict(orient="records")