import os
from dotenv import load_dotenv

from llama_index.core.readers.file.base import SimpleDirectoryReader
from llama_index.core import VectorStoreIndex, StorageContext, load_index_from_storage, Settings
from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.llms.openai import OpenAI
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.service_context import ServiceContext
import faiss

load_dotenv()

VECTOR_DB_DIR = "vector_db"
os.makedirs(VECTOR_DB_DIR, exist_ok=True)

# Embedding model (local, 384 dims)
embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

def create_faiss_index():
    return faiss.IndexFlatL2(384)

def ingest_documents(documents):
    global vector_index
    splitter = SentenceSplitter(chunk_size=512, chunk_overlap=50)
    nodes = []
    for doc in documents:
        nodes.extend(splitter.get_nodes_from_documents([doc]))

    # Create a new FAISS index and vector store
    faiss_index = create_faiss_index()
    vector_store = FaissVectorStore(faiss_index)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # Build the index from documents
    vector_index = VectorStoreIndex.from_documents(
        nodes,
        storage_context=storage_context,
        embed_model=embed_model,
    )

    # Persist the full index (nodes + vector store)
    vector_index.storage_context.persist(persist_dir=VECTOR_DB_DIR)

    # Manually persist FAISS index (optional, but ensures index.faiss is written)
    import faiss
    faiss.write_index(faiss_index, os.path.join(VECTOR_DB_DIR, "index.faiss"))

    return vector_index

def load_vector_index():
    # Loads both the FAISS index and the document nodes
    storage_context = StorageContext.from_defaults(persist_dir=VECTOR_DB_DIR)
    try:
        return load_index_from_storage(storage_context, embed_model=embed_model)
    except FileNotFoundError:
        return None  # No index yet


def build_query_engine(retriever):
    llm = OpenAI(
        temperature=0.4,
        model="gpt-4.1-nano-2025-04-14", #  gpt-4o-2024-11-20
        max_tokens=10000,
        api_base=os.getenv("OPENROUTER_BASE_URL"),
        api_key=os.getenv("OPENROUTER_API_KEY")
    )

    return RetrieverQueryEngine.from_args(
        retriever=retriever,
        llm=llm  # <-- pass as a keyword argument, not called
    )


