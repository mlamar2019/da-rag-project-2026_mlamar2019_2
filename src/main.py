from fastapi import FastAPI, Depends
from pydantic import BaseModel

from ingestion import VectorStoreManager


app = FastAPI(title="Mini Wikipedia RAG API", version="0.1.0")

# Global vector store instance
_vector_store: VectorStoreManager | None = None


def get_vector_store() -> VectorStoreManager:
    """Get or initialize the global vector store manager."""
    global _vector_store
    if _vector_store is None:
        _vector_store = VectorStoreManager()
    return _vector_store


class EchoRequest(BaseModel):
    message: str


class EchoResponse(BaseModel):
    message: str


class VectorDBStatus(BaseModel):
    initialized: bool
    document_count: int
    index_info: str | None


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/echo", response_model=EchoResponse)
def echo(payload: EchoRequest) -> EchoResponse:
    return EchoResponse(message=payload.message)


@app.get("/vectordb/status", response_model=VectorDBStatus)
def get_vectordb_status(store: VectorStoreManager = Depends(get_vector_store)) -> VectorDBStatus:
    """Get current vector database status."""
    status = store.get_status()
    return VectorDBStatus(**status)


@app.get("/vectordb/query")
def query_vectordb(
    q: str,
    top_k: int = 5,
    store: VectorStoreManager = Depends(get_vector_store),
) -> list[dict]:
    """Query the vector database for similar documents."""
    return store.query(q, top_k=top_k)
