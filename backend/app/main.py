from fastapi import FastAPI
from app.routes import query, upload, summarize, charts, embedding
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Or specify your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(upload.router)
app.include_router(query.router)
app.include_router(summarize.router)
app.include_router(charts.router)
app.include_router(embedding.router)

from app.routes.image import router as image_router
app.include_router(image_router)