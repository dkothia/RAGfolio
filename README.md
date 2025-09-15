# RAGfolio

RAGfolio is a Retrieval-Augmented Generation (RAG) portfolio project that enables users to upload PDFs, images, or URLs, extract and index their content (including OCR for scanned documents), and query the ingested data using natural language prompts. The backend is built with FastAPI and integrates with FAISS for vector search and LlamaIndex for document management and LLM-powered responses. The frontend is built with React.

---

## Features

- **Upload PDFs, Images, or URLs:**  
  Supports uploading of PDF files (including scanned documents), images, or web URLs for ingestion.

- **OCR Support:**  
  Automatically extracts text from scanned PDFs and images using Tesseract OCR.

- **Embedded Image Extraction:**  
  Extracts and OCRs embedded images from PDFs.

- **Vector Search with FAISS:**  
  All extracted text is embedded and indexed using FAISS for fast similarity search.

- **Natural Language Query:**  
  Users can enter prompts to query the ingested data and receive LLM-generated answers.

- **API Key Security:**  
  All endpoints require an API key (`newyorkisgreat`) sent via the `X-API-Key` header.

- **Rate Limiting:**  
  Prevents abuse with configurable rate limits.

---

## Backend Setup

### Prerequisites

- Python 3.9+
- [Tesseract OCR](https://github.com/tesseract-ocr/tesseract) installed and in PATH
- [Poppler](https://github.com/oschwartz10612/poppler-windows) installed and in PATH (for PDF image conversion)
- (Recommended) Create and activate a Python virtual environment:
  ```sh
  python3 -m venv .venv
  source .venv/bin/activate
  ```

### Install Dependencies

```sh
pip install -r requirements.txt
```

### Environment Variables

Set any required environment variables in a `.env` file if needed.

### Run the Backend

```sh
uvicorn app.main:app --reload
```

The API will be available at [http://127.0.0.1:8000](http://127.0.0.1:8000).

---

## API Endpoints

- `POST /upload`  
  Upload a PDF, image, or URL for ingestion. Requires `X-API-Key` header.

- `POST /query`  
  Query the ingested data with a prompt. Requires `X-API-Key` header.  
  Body: `{ "question": "your prompt" }`

- `POST /image-ocr`  
  Query extracted image text with a prompt. Requires `X-API-Key` header.

- `GET /embedding`  
  (Debug) View chunks and FAISS vectors.

---

## Frontend Setup

### Prerequisites

- Node.js 18+

### Install and Run

```sh
cd frontend
npm install
npm start
```

The React app will be available at [http://localhost:3000](http://localhost:3000).

### Configuration

Set the backend API URL in `frontend/.env`:
```
REACT_APP_API_URL=http://127.0.0.1:8000
```

---

## Usage

1. **Start the backend and frontend servers.**
2. **Open the frontend in your browser.**
3. **Enter your API key (`newyorkisgreat`).**
4. **Upload a PDF, image, or enter a URL.**
5. **Enter a prompt to query the ingested data.**
6. **View the answer in the response box.**

---

## Notes

- Make sure Tesseract and Poppler are installed and accessible in your system PATH.
- The API key is required for all protected endpoints.
- For best results, use high-quality scans for OCR.

---

## License

MIT License

---