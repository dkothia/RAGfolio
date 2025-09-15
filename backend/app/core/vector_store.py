import os
from llama_index.core import StorageContext, load_index_from_storage
from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.service_context import ServiceContext

VECTOR_DB_DIR = "vector_db"

embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

def load_faiss_index():
    vector_store = FaissVectorStore.from_persist_dir(VECTOR_DB_DIR)
    storage_context = StorageContext.from_defaults(
        persist_dir=VECTOR_DB_DIR,
        vector_store=vector_store
    )
    return load_index_from_storage(storage_context, embed_model=embed_model)

