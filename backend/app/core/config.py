import os

class Settings:
    AWS_REGION = "us-west-2"
    S3_BUCKET_NAME = "ragfolio"
    FAISS_INDEX_PATH = "faiss_store"
    AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
    AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")

    # RAGfolio-specific paths
    FAISS_INDEX_PATH = "faiss_store"
    TEMP_DIR = "temp_docs"

    # OpenRouter
    OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
    OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL")
settings = Settings()
