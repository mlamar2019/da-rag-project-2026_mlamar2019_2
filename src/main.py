from fastapi import FastAPI, Depends
from pydantic import BaseModel

from ingestion import VectorStoreManager, DEFAULT_PASSAGES_SOURCE
from ailab.utils.azure import get_ailab_auth_status


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


class IngestRequest(BaseModel):
    source: str = DEFAULT_PASSAGES_SOURCE
    limit: int | None = 100


class IngestResponse(BaseModel):
    ingested_count: int
    source: str
    status: VectorDBStatus


class AuthStatusResponse(BaseModel):
    authenticated: bool
    auth_source: str | None
    scope: str
    error: str | None = None


class QueryRequest(BaseModel):
    query: str
    top_k: int = 5


class QueryEmbeddingResponse(BaseModel):
    query: str
    embedding: list[float]
    dimensions: int


class RetrievedDocument(BaseModel):
    text: str
    score: float | None = None
    metadata: dict


class QueryResponse(BaseModel):
    query: str
    top_k: int
    embedding: list[float]
    results: list[RetrievedDocument]


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/echo", response_model=EchoResponse)
def echo(payload: EchoRequest) -> EchoResponse:
    return EchoResponse(message=payload.message)


@app.get("/auth/status", response_model=AuthStatusResponse)
def auth_status() -> AuthStatusResponse:
    """Return whether the server currently has usable auth material available."""
    return AuthStatusResponse(**get_ailab_auth_status())


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


@app.post("/vectordb/query/embedding", response_model=QueryEmbeddingResponse)
def embed_query(
    payload: QueryRequest,
    store: VectorStoreManager = Depends(get_vector_store),
) -> QueryEmbeddingResponse:
    """Generate an embedding for a user query."""
    embedding = store.generate_query_embedding(payload.query)
    return QueryEmbeddingResponse(
        query=payload.query,
        embedding=embedding,
        dimensions=len(embedding),
    )


@app.post("/vectordb/query", response_model=QueryResponse)
def query_vectordb_post(
    payload: QueryRequest,
    store: VectorStoreManager = Depends(get_vector_store),
) -> QueryResponse:
    """Embed a query, run similarity search, and return retrieved documents."""
    embedding = store.generate_query_embedding(payload.query)
    results = store.retrieve(
        query_str=payload.query,
        top_k=payload.top_k,
        query_embedding=embedding,
    )
    return QueryResponse(
        query=payload.query,
        top_k=payload.top_k,
        embedding=embedding,
        results=[RetrievedDocument(**result) for result in results],
    )


@app.post("/vectordb/ingest", response_model=IngestResponse)
def ingest_vectordb(
    payload: IngestRequest,
    store: VectorStoreManager = Depends(get_vector_store),
) -> IngestResponse:
    """Ingest documents from a source parquet file into vector DB."""
    ingested_count = store.ingest_from_source(source=payload.source, limit=payload.limit)
    status = VectorDBStatus(**store.get_status())
    return IngestResponse(
        ingested_count=ingested_count,
        source=payload.source,
        status=status,
    )

