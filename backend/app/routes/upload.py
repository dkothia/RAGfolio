from fastapi import APIRouter, File, UploadFile, Form, BackgroundTasks, Depends, HTTPException, Security, Request
from fastapi.responses import JSONResponse
from fastapi.security.api_key import APIKeyHeader
from typing import Optional
import os, shutil, uuid, requests
from bs4 import BeautifulSoup

from app.core.config import settings
from app.core.vector_store import load_faiss_index
from app.core.rag_engine import ingest_documents
from app.core.aws_utils import upload_file_to_s3, download_file_from_s3

from llama_index.core import SimpleDirectoryReader, download_loader
from llama_index.readers.web import SimpleWebPageReader
from llama_index.core.schema import Document
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from fastapi import FastAPI

from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from PIL import Image
from pdf2image import convert_from_path
import pytesseract
import io
import fitz  # PyMuPDF

API_KEY = "newyorkisgreat"
api_key_header = APIKeyHeader(name="X-API-Key")

router = APIRouter()
limiter = Limiter(key_func=get_remote_address)
app = FastAPI()
app.state.limiter = limiter
app.add_exception_handler(429, _rate_limit_exceeded_handler)

embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

def verify_api_key(api_key: str = Security(api_key_header)):
    if api_key != API_KEY:
        raise HTTPException(status_code=403, detail="Invalid API Key")

def get_one_level_links(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')

    base_url = "/".join(url.split("/")[:3])
    links = set()

    for a_tag in soup.find_all('a', href=True):
        href = a_tag['href']
        if href.startswith("http"):
            links.add(href)
        elif href.startswith("/"):
            links.add(base_url + href)

    # Always include the main page itself
    links.add(url)
    return list(links)

def extract_images_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    image_texts = []
    for page_index in range(len(doc)):
        page = doc[page_index]
        images = page.get_images(full=True)
        for img_index, img in enumerate(images):
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            img_pil = Image.open(io.BytesIO(image_bytes))
            ocr_text = pytesseract.image_to_string(img_pil)
            if ocr_text.strip():
                image_texts.append(Document(text=ocr_text.strip()))
    return image_texts

@router.post("/upload")
@limiter.limit("5/minute")
async def upload(
    request: Request,
    background_tasks: BackgroundTasks,
    pdf: Optional[UploadFile] = File(None),
    url: Optional[str] = Form(None),
    image: Optional[UploadFile] = File(None),  # <-- Add image field
    api_key: str = Depends(verify_api_key)
):
    documents = []
    os.makedirs(settings.TEMP_DIR, exist_ok=True)

    if pdf:
        file_id = str(uuid.uuid4())
        local_path = os.path.join(settings.TEMP_DIR, f"{file_id}_{pdf.filename}")
        with open(local_path, "wb") as f:
            shutil.copyfileobj(pdf.file, f)

        s3_key = f"ragfolio_uploads/{file_id}_{pdf.filename}"
        if not upload_file_to_s3(local_path, s3_key):
            return JSONResponse(status_code=500, content={"message": "S3 upload failed"})

        ingest_path = os.path.join(settings.TEMP_DIR, f"s3copy_{pdf.filename}")
        if not download_file_from_s3(s3_key, ingest_path):
            return JSONResponse(status_code=500, content={"message": "S3 download failed"})

        # Step 1: Try to extract selectable text and embedded images
        documents = []
        reader = SimpleDirectoryReader(input_files=[ingest_path])
        text_docs = reader.load_data()
        if text_docs:
            documents.extend(text_docs)
            print(f"Selectable text extracted: {len(text_docs)} chunks.")
        else:
            print("No selectable text found.")

        try:
            image_docs = extract_images_from_pdf(ingest_path)
            if image_docs:
                documents.extend(image_docs)
                print(f"Embedded images extracted: {len(image_docs)} chunks.")
            else:
                print("No embedded images found.")
        except Exception as e:
            print(f"PDF embedded image extraction/OCR failed: {e}")

        # Step 2: If no text or images, run OCR on scanned pages
        if not documents:
            print("No text or embedded images found, running OCR on scanned PDF pages...")
            try:
                page_images = convert_from_path(ingest_path)
                print(f"Total pages converted to images: {len(page_images)}")
                for img in page_images:
                    ocr_text = pytesseract.image_to_string(img)
                    if ocr_text.strip():
                        # Optionally split into chunks here
                        chunk_size = 500
                        ocr_chunks = [ocr_text[i:i+chunk_size] for i in range(0, len(ocr_text), chunk_size)]
                        for chunk in ocr_chunks:
                            documents.append(Document(text=chunk.strip()))
                print("OCR on scanned PDF pages complete.")
            except Exception as e:
                print(f"PDF page image OCR failed: {e}")

    if url:
        #all_links = get_one_level_links(url)
        all_links = [url]  # For now, just use the main URL
        web_reader = SimpleWebPageReader()
        documents += web_reader.load_data(urls=all_links)

    if image:
        image_bytes = await image.read()
        img = Image.open(io.BytesIO(image_bytes))
        extracted_text = pytesseract.image_to_string(img)
        if extracted_text.strip():
            from llama_index.core.schema import Document
            documents.append(Document(text=extracted_text.strip()))
        else:
            return JSONResponse(status_code=400, content={"message": "No text found in image."})

    if not documents:
        return JSONResponse(status_code=400, content={"message": "No valid input provided."})

    # Use background task for ingestion
    background_tasks.add_task(ingest_documents, documents)
    #ingest_documents(documents)

    # Wait/check for index.faiss creation (simple polling, max 10s)
    import time
    index_path = os.path.join("vector_db", "index.faiss")
    for _ in range(20):  # 20 x 0.5s = 10s max
        if os.path.isfile(index_path):
            return {"message": "Upload and ingestion complete. index.faiss created."}
        time.sleep(0.5)

    return JSONResponse(
        status_code=202,
        content={"message": "Upload received, ingestion started in background. index.faiss not yet created."}
    )


