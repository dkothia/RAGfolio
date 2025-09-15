from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from app.core.vector_store import load_faiss_index
from app.core.rag_engine import build_query_engine

router = APIRouter()

class ImageQueryRequest(BaseModel):
    prompt: str

class OCRResponse(BaseModel):
    text: str
    summary: str

@router.post("/image-ocr", response_model=OCRResponse)
async def process_image_ocr(req: ImageQueryRequest):
    try:
        # Load FAISS index
        index = load_faiss_index()
        retriever = index.as_retriever(similarity_top_k=1)
        query_engine = build_query_engine(retriever)

        # Retrieve relevant image text document(s) using user prompt
        nodes = retriever.retrieve(req.prompt)
        if not nodes:
            raise HTTPException(status_code=404, detail="No relevant image text found in vector database.")

        extracted_text = nodes[0].get_content()

        # Use user prompt and extracted text directly
        output = f"Summarize based on \n User prompt: {req.prompt}\nExtracted text: {extracted_text.strip()}"

        return OCRResponse(text=output, summary=str(query_engine.query(output)).strip())
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"OCR processing failed: {str(e)}")